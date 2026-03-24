from typing import TYPE_CHECKING

from mqns.entity.memory import MemoryQubit, QuantumMemory, QubitState
from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.message import PurifResponseMsg, PurifSolicitMsg
from mqns.network.network import QuantumNetwork
from mqns.simulator import Simulator
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class ForwarderPurifProc:
    """
    Part of ``Forwarder`` logic related to purification procedure.
    """

    fw: "Forwarder"
    simulator: Simulator
    epr_type: type[Entanglement]
    network: QuantumNetwork
    node: QNode
    memory: QuantumMemory

    def install(self, fw: "Forwarder"):
        self.fw = fw
        self.simulator = fw.simulator
        self.epr_type = fw.epr_type
        self.network = fw.network
        self.node = fw.node
        self.memory = fw.memory

    def start(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry, partner: QNode):
        """
        Initiate purification protocol.

        Args:
            mq0: first memory qubit, which would be kept if purification succeeds.
            mq1: second memory qubit, which is consumed during purification.
            fib_entry: FIB entry.
            partner: quantum node with which entanglements are shared.
        """
        # read qubits to set fidelity at this time
        _, epr0 = self.memory.read(mq0.addr, has=self.epr_type, set_fidelity=True)
        _, epr1 = self.memory.read(mq1.addr, has=self.epr_type, set_fidelity=True, remove=True)

        log.debug(
            f"{self.node}: request purif qubit {mq0.addr} (F={epr0.fidelity}) and "
            + f"{mq1.addr} (F={epr1.fidelity}) with partner {partner.name}"
        )

        mq0.state = QubitState.PENDING
        self.fw.release_qubit(mq1)

        # send purif_solicit to partner
        msg: PurifSolicitMsg = {
            "cmd": "PURIF_SOLICIT",
            "path_id": fib_entry.path_id,
            "purif_node": self.node.name,
            "partner": partner.name,
            "epr": epr0.name,
            "measure_epr": epr1.name,
            "round": mq0.purif_rounds,
        }
        self.fw.send_msg(partner, msg, fib_entry)

    def handle_solicit(self, msg: PurifSolicitMsg, fib_entry: FibEntry):
        """
        Process a PURIF_SOLICIT message from primary node as part of the purification protocol.

        1. Retrieve the target and auxiliary qubits from memory and verify their states.
        2. Attempt purification.
        3. If successful, update the EPR and send a PURIF_RESPONSE with result=True.
        4. Otherwise, mark both qubits for release and reply with result=False.

        Args:
            msg: Message containing purification parameters and EPR names.
            fib_entry: FIB entry associated with path_id in the message.

        Notes:
            If EPR purification succeeds, if the qubit has completed the required rounds of purifications,
            it may immediately become eligible and thus available for swaps or end-to-end consumption,
            even if the PURIF_RESPONSE message has not arrived at the primary node.
        """
        # mq0 is the "kept" memory whose fidelity would be increased if purification succeeds
        # mq1 is the "measured" memory that is consumed during purification
        mq0, epr0 = self.memory.read(msg["epr"], has=self.epr_type, set_fidelity=True)
        mq1, epr1 = self.memory.read(msg["measure_epr"], has=self.epr_type, set_fidelity=True, remove=True)
        # TODO: handle the exception case when an EPR is decohered and not found in memory

        for mq in (mq0, mq1):
            assert mq.state == QubitState.PURIF
            assert mq.purif_rounds == msg["round"]

        assert msg["partner"] == self.node.name
        primary = self.network.get_node(msg["purif_node"])
        log.debug(
            f"{self.node}: perform purif qubit {mq0.addr} (F={epr0.fidelity}) and "
            + f"{mq1.addr} (F={epr1.fidelity}) for round {1 + mq0.purif_rounds} with primary {primary.name}"
        )

        # perform purification between EPRs
        result = epr0.purify(epr1, now=self.simulator.tc)
        log.debug(
            f"{self.node}: purif {'succeeded' if result else 'failed'} on qubit {mq0.addr} (F={epr0.fidelity}) "
            + f"for round {1 + mq0.purif_rounds} with primary {primary.name}"
        )

        if result:
            self.memory.write(mq0.addr, epr0, replace=True)
            self.fw.cnt.increment_n_purif(mq0.purif_rounds)
            mq0.purif_rounds += 1
            mq0.state = QubitState.PURIF
            self.fw.qubit_is_purif(mq0, fib_entry, primary)
        else:
            # in case of purification failure, release mq0
            self.fw.release_qubit(mq0, need_remove=True)

        # release mq1; destructive reading is already performed
        self.fw.release_qubit(mq1)

        # send response message
        resp: PurifResponseMsg = {
            **msg,
            "cmd": "PURIF_RESPONSE",
            "result": result,
        }
        self.fw.send_msg(primary, resp, fib_entry)

    def handle_response(self, msg: PurifResponseMsg, fib_entry: FibEntry):
        """
        Process a PURIF_RESPONSE message indicating the outcome of a purification attempt.

        If the purification succeeded:

        1. Update the EPR.
        2. Increment the qubit's purification round counter.
        3. Allow the qubit to re-enter the purification process.

        If the purification failed:

        1. Release the qubit.

        Args:
            msg: Response message containing the result and identifiers of the purified EPRs.
            fib_entry: FIB entry associated with path_id in the message.

        """
        qubit, epr = self.memory.read(msg["epr"], has=self.epr_type)
        # TODO: handle the exception case when an EPR is decohered and not found in memory

        result = msg["result"]
        log.debug(
            f"{self.node}: purif {'succeeded' if result else 'failed'} on qubit {qubit.addr} (F={epr.fidelity}) "
            + f"for round {1 + qubit.purif_rounds} with partner {msg['partner']}"
        )

        if not result:  # purif failed
            self.fw.release_qubit(qubit, need_remove=True)
            return

        # purif succeeded
        self.memory.write(qubit.addr, epr, replace=True)
        self.fw.cnt.increment_n_purif(qubit.purif_rounds)
        qubit.purif_rounds += 1
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, self.network.get_node(msg["partner"]))
