from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.network.fw.fib import FibPath
from mqns.network.fw.fw_module import ForwarderModule, fw_signaling_cmd_handler
from mqns.network.fw.message import PurifResponseMsg, PurifSolicitMsg
from mqns.utils import unwrap_cast


class ForwarderPurifProc(ForwarderModule):
    """
    Part of ``Forwarder`` logic related to purification procedure.
    """

    def start(self, mq0: MemoryQubit, mq1: MemoryQubit, fp: FibPath, partner: QNode):
        """
        Initiate purification protocol.

        Args:
            mq0: first memory qubit, which would be kept if purification succeeds.
            mq1: second memory qubit, which is consumed during purification.
            fp: FIB path entry.
            partner: quantum node with which entanglements are shared.
        """
        # read qubits to set fidelity at this time
        now = self.simulator.tc
        _, epr0 = self.memory.read(mq0.addr, has=self.epr_type)
        _, epr1 = self.memory.read(mq1.addr, has=self.epr_type, remove=True)
        epr0.apply_store_decays(now)
        epr1.apply_store_decays(now)

        self.log_debug(
            "request purif qubit %s (F=%s) and %s (F=%s) with partner %s",
            mq0.addr,
            epr0.fidelity,
            mq1.addr,
            epr1.fidelity,
            partner.name,
        )

        # send purif_solicit to partner
        msg: PurifSolicitMsg = {
            "cmd": "PURIF_SOLICIT",
            "path_id": fp.path_id,
            "purif_node": self.node.name,
            "partner": partner.name,
            "key0": unwrap_cast(mq0.partner).key,
            "key1": unwrap_cast(mq1.partner).key,
            "round": mq0.purif_rounds,
        }
        self.send_msg(partner, msg, fp)

        mq0.state = QubitState.PENDING
        self.fw.release_qubit(mq1)

    @fw_signaling_cmd_handler("PURIF_SOLICIT")
    def handle_solicit(self, msg: PurifSolicitMsg, fp: FibPath):
        """
        Process a PURIF_SOLICIT message from primary node as part of the purification protocol.

        1. Retrieve the target and auxiliary qubits from memory and verify their states.
        2. Attempt purification.
        3. If successful, update the EPR and send a PURIF_RESPONSE with result=True.
        4. Otherwise, mark both qubits for release and reply with result=False.

        Notes:
            If EPR purification succeeds, if the qubit has completed the required rounds of purifications,
            it may immediately become eligible and thus available for swaps or end-to-end consumption,
            even if the PURIF_RESPONSE message has not arrived at the primary node.
        """
        # mq0 is the "kept" memory whose fidelity would be increased if purification succeeds
        # mq1 is the "measured" memory that is consumed during purification
        mq0, epr0 = self.memory.read(msg["key0"], has=self.epr_type)
        mq1, epr1 = self.memory.read(msg["key1"], has=self.epr_type, remove=True)
        # TODO: handle the exception case when an EPR is decohered and not found in memory
        p_key0 = unwrap_cast(mq0.partner).key
        p_key1 = unwrap_cast(mq1.partner).key

        for mq in (mq0, mq1):
            assert mq.state is QubitState.PURIF, f"unexpected state {mq.state}"
            assert mq.purif_rounds == msg["round"]

        assert msg["partner"] == self.node.name
        primary = self.network.get_node(msg["purif_node"])
        self.log_debug(
            "perform purif qubit %s (F=%s) and %s (F=%s) for round %s with primary %s",
            mq0.addr,
            epr0.fidelity,
            mq1.addr,
            epr1.fidelity,
            1 + mq0.purif_rounds,
            primary.name,
        )

        # perform purification between EPRs
        result = epr0.purify(epr1, now=self.simulator.tc)
        self.log_debug(
            "purif %s on qubit %s (F=%s) for round %s with primary %s",
            "succeeded" if result else "failed",
            mq0.addr,
            epr0.fidelity,
            1 + mq0.purif_rounds,
            primary.name,
        )

        if result:
            self.memory.write(mq0.addr, epr0, replace=True)
            self.fw_cnt.increment_n_purif(mq0.purif_rounds)
            mq0.purif_rounds += 1
            mq0.state = QubitState.PURIF
            self.fw.qubit_is_purif(mq0, fp, primary)
        else:
            # in case of purification failure, release mq0
            self.fw.release_qubit(mq0, need_remove=True)

        # release mq1; destructive reading is already performed
        self.fw.release_qubit(mq1)

        # send response message
        resp: PurifResponseMsg = {
            **msg,
            "cmd": "PURIF_RESPONSE",
            "key0": p_key0,
            "key1": p_key1,
            "result": result,
        }
        self.send_msg(primary, resp, fp)

    @fw_signaling_cmd_handler("PURIF_RESPONSE")
    def handle_response(self, msg: PurifResponseMsg, fp: FibPath):
        """
        Process a PURIF_RESPONSE message indicating the outcome of a purification attempt.

        If the purification succeeded:

        1. Update the EPR.
        2. Increment the qubit's purification round counter.
        3. Allow the qubit to re-enter the purification process.

        If the purification failed:

        1. Release the qubit.
        """
        qubit, epr = self.memory.read(msg["key0"], has=self.epr_type)
        # TODO: handle the exception case when an EPR is decohered and not found in memory

        result = msg["result"]
        self.log_debug(
            "purif %s on qubit %s (F=%s) for round %s with partner %s",
            "succeeded" if result else "failed",
            qubit.addr,
            epr.fidelity,
            1 + qubit.purif_rounds,
            msg["partner"],
        )

        if not result:  # purif failed
            self.fw.release_qubit(qubit, need_remove=True)
            return

        # purif succeeded
        self.memory.write(qubit.addr, epr, replace=True)
        self.fw_cnt.increment_n_purif(qubit.purif_rounds)
        qubit.purif_rounds += 1
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fp, self.network.get_node(msg["partner"]))
