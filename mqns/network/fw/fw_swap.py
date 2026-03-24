from typing import TYPE_CHECKING, cast

from mqns.entity.memory import MemoryQubit, QuantumMemory, QubitState
from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.message import SwapUpdateMsg
from mqns.network.fw.mux import MuxScheme
from mqns.network.network import QuantumNetwork
from mqns.simulator import Simulator
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class ForwarderSwapProc:
    """
    Part of ``Forwarder`` logic related to swapping procedure.
    """

    fw: "Forwarder"
    simulator: Simulator
    epr_type: type[Entanglement]
    network: QuantumNetwork
    node: QNode
    memory: QuantumMemory
    mux: MuxScheme

    def __init__(self, *, ps: float):
        self.ps = ps
        """Probability of successful entanglement swapping."""
        assert 0.0 <= self.ps <= 1.0

        self.waiting_su: dict[int, tuple[SwapUpdateMsg, FibEntry]] = {}
        """
        SwapUpdates received prior to QubitEntangledEvent.

        * Key: MemoryQubit addr.
        * Value: SwapUpdateMsg and FibEntry.
        """

        self.parallel_swappings: dict[str, tuple[Entanglement, Entanglement, Entanglement]] = {}
        """
        Records for potential parallel swappings.
        See ``_su_parallel`` method.
        """

        self.remote_swapped_eprs: dict[str, Entanglement] = {}
        """
        EPRs that have been swapped remotely but the SwapUpdateMsg have not arrived.
        Each key is an EPR name; each value is the EPR.

        When a remote forwarder performs a swapping in which this node is either src or dst of the new EPR,
        it deposits the swapped EPR here and transmits the corresponding SwapUpdateMsg.
        Upon receiving the SwapUpdateMsg, the local forwarder pops the EPR.

        XXX Current approach assumes cchannels do not have packet loss.
        """

    def _deposit_remote_swapped_epr(self, target: QNode, epr: Entanglement):
        target.get_app(type(self.fw)).swap.remote_swapped_eprs[epr.name] = epr

    def install(self, fw: "Forwarder"):
        self.fw = fw
        self.simulator = fw.simulator
        self.epr_type = fw.epr_type
        self.network = fw.network
        self.node = fw.node
        self.memory = fw.memory
        self.mux = fw.mux

    def start(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry):
        """
        Start swapping between two qubits at an intermediate node.
        These qubits must be in ELIGIBLE state and come from different qchannels.
        Partners are notified with SWAP_UPDATE messages.
        """
        assert mq0.addr != mq1.addr
        assert mq0.state == QubitState.ELIGIBLE
        assert mq1.state == QubitState.ELIGIBLE

        # Read both qubits and remove them from memory.
        #
        # One of these qubits must be entangled with a partner node to the left of the current node.
        # This is determined by epr.dst==self.node condition, because LinkLayer establishes elementary
        # entanglements from left to right, and swapping maintains this condition.
        # This qubit and related objects are assigned to prev_* variables.
        #
        # Likewise, the other qubit entangled with a partner node to the right is assigned to next_*.
        prev_tuple: tuple[QNode, MemoryQubit, Entanglement] | None = None
        next_tuple: tuple[QNode, MemoryQubit, Entanglement] | None = None
        for addr in (mq0.addr, mq1.addr):
            qubit, epr = self.memory.read(addr, has=self.epr_type, remove=True)
            if epr.dst == self.node:
                assert epr.src is not None
                prev_tuple = epr.src, qubit, epr
            elif epr.src == self.node:
                assert epr.dst is not None
                next_tuple = epr.dst, qubit, epr
            else:
                raise Exception(f"Unexpected: swapping EPRs {mq0} x {mq1}")

        # Make sure both partners are found.
        assert prev_tuple is not None
        assert next_tuple is not None
        _, prev_qubit, prev_epr = prev_tuple
        _, next_qubit, next_epr = next_tuple

        # Save ch_index metadata field onto elementary EPR.
        if not prev_epr.orig_eprs:
            prev_epr.ch_index = fib_entry.own_idx - 1
        if not next_epr.orig_eprs:
            next_epr.ch_index = fib_entry.own_idx

        # Attempt the swap.
        new_epr = Entanglement.swap(prev_epr, next_epr, now=self.simulator.tc, ps=self.ps)
        log.debug(f"{self.node}: SWAP {'SUCC' if new_epr else 'FAILED'} | {prev_qubit} x {next_qubit} = {new_epr}")

        if new_epr is not None:  # swapping succeeded
            self.fw.cnt.n_swapped_s += 1

            # Inform multiplexing scheme.
            self.mux.swapping_succeeded(prev_epr, next_epr, new_epr)

        for (a_partner, a_qubit, a_epr), (b_partner, _, b_epr) in ((prev_tuple, next_tuple), (next_tuple, prev_tuple)):
            if new_epr is not None:
                # Keep records to support potential parallel swapping.
                _, a_rank = fib_entry.find_index_and_swap_rank(a_partner.name)
                if fib_entry.own_swap_rank == a_rank:
                    self.parallel_swappings[a_epr.name] = (a_epr, b_epr, new_epr)

                # Deposit swapped EPR at the partner.
                self._deposit_remote_swapped_epr(a_partner, new_epr)

            # Send SWAP_UPDATE to the partner.
            su_msg: SwapUpdateMsg = {
                "cmd": "SWAP_UPDATE",
                "path_id": fib_entry.path_id,
                "swapping_node": self.node.name,
                "partner": b_partner.name,
                "epr": a_epr.name,
                "new_epr": None if new_epr is None else new_epr.name,
            }
            self.fw.send_msg(a_partner, su_msg, fib_entry)

            # Release old qubit.
            self.fw.release_qubit(a_qubit)

    def pop_waiting_su(self, qubit: MemoryQubit):
        su_args = self.waiting_su.pop(qubit.addr, None)
        if (
            qubit.state != QubitState.RELEASE  # qubit was released due to uninstalled path
            and su_args
        ):
            self.handle_update(*su_args)

    def handle_update(self, msg: SwapUpdateMsg, fib_entry: FibEntry):
        """
        Process an SWAP_UPDATE signaling message.
        It may either update local qubit state or release decohered pairs.

        If QubitEntangledEvent for the qubit has not been processed, the SwapUpdate is buffered
        in ``self.waiting_su`` and will be re-tried after processing the QubitEntangledEvent.
        This may happen, for example in S-R-D linear topology, when node R performs a swap as soon as
        it is notified about R-D entanglement, before D is notified.
        This cannot be resolved with ``Event.priority`` mechanism because R and D may be notified
        at different times depending on the link architecture.

        Args:
            msg: The SWAP_UPDATE message.
            fib_entry: FIB entry associated with path_id in the message.

        """
        if not self.node.timing.is_internal():
            log.debug(f"{self.node}: INT phase is over -> stop swaps")
            return

        _, sender_rank = fib_entry.find_index_and_swap_rank(msg["swapping_node"])
        if fib_entry.own_swap_rank < sender_rank:
            log.debug(f"### {self.node}: VERIFY -> rcvd SU from higher-rank node")
            return

        new_epr_name = msg["new_epr"]
        new_epr = None if new_epr_name is None else self.remote_swapped_eprs.pop(new_epr_name)

        epr_name = msg["epr"]
        qubit_pair = self.memory.read(epr_name)
        if qubit_pair is not None:
            qubit, _ = qubit_pair
            if qubit.state == QubitState.ENTANGLED0:
                if new_epr is not None:
                    self.remote_swapped_eprs[cast(str, new_epr_name)] = new_epr
                self.waiting_su[qubit.addr] = (msg, fib_entry)
                return
            self.parallel_swappings.pop(epr_name, None)
            self._su_sequential(msg, fib_entry, qubit, new_epr, maybe_purif=(fib_entry.own_swap_rank > sender_rank))
        elif fib_entry.own_swap_rank == sender_rank and epr_name in self.parallel_swappings:
            self._su_parallel(msg, fib_entry, new_epr)
        else:
            log.debug(f"### {self.node}: EPR {epr_name} decohered during SU transmissions")

    def _su_sequential(
        self,
        msg: SwapUpdateMsg,
        fib_entry: FibEntry,
        qubit: MemoryQubit,
        new_epr: Entanglement | None,
        maybe_purif: bool,
    ):
        """
        Process SWAP_UPDATE message where the local MemoryQubit still exists.
        This means the swapping was performed sequentially and local MemoryQubit has not decohered.

        Args:
            maybe_purif: whether the new EPR may enter PURIF state.
                         Set to True if own rank is higher than sender rank.
        """
        if (
            new_epr is None  # swapping failed
            or new_epr.decohere_time <= self.simulator.tc  # oldest pair decohered
        ):
            if new_epr:
                log.debug(f"{self.node}: NEW EPR {new_epr} decohered during SU transmissions")
            # Inform LinkLayer that the memory qubit has been released.
            self.fw.release_qubit(qubit, need_remove=True)
            return

        # Update old EPR with new EPR (fidelity and partner).
        self.memory.write(qubit.addr, new_epr, replace=True)

        if maybe_purif:
            # If own rank is higher than sender rank but lower than new partner rank,
            # it is our turn to purify the qubit and progress toward swapping.
            qubit.purif_rounds = 0
            qubit.state = QubitState.PURIF
            partner = self.network.get_node(msg["partner"])
            self.fw.qubit_is_purif(qubit, fib_entry, partner)

    def _su_parallel(self, msg: SwapUpdateMsg, fib_entry: FibEntry, new_epr: Entanglement | None):
        """
        Process SWAP_UPDATE message during parallel swapping.
        """
        shared_epr, other_epr, my_new_epr = self.parallel_swappings.pop(msg["epr"])
        _ = shared_epr

        # safety in statistical mux to avoid conflictual swappings on different paths
        if self.mux.su_parallel_has_conflict(my_new_epr, msg["path_id"]):
            self.fw.cnt.n_swap_conflict += 1
            return

        # msg["swapping_node"] is the node that performed swapping and sent this message.
        # Assuming swapping_node is to the right of own node, various nodes and EPRs are as follows:
        #
        # destination-------own--------swapping_node----partner
        #      |             |~~shared_epr~~|            |
        #      |~~other_epr~~|              |            |
        #      |~~~~~~~~~~my_new_epr~~~~~~~~|            |
        #      |             |~~~~~~~~~~new_epr~~~~~~~~~~|
        #      |~~~~~~~~~~~~~~~merged_epr~~~~~~~~~~~~~~~~|

        if (
            new_epr is None  # swapping failed
            or new_epr.decohere_time <= self.simulator.tc  # oldest pair decohered
        ):
            # Determine the "destination".
            if other_epr.dst == self.node:  # destination is to the left of own node
                destination = other_epr.src
            else:  # destination is to the right of own node
                destination = other_epr.dst
            assert destination is not None

            # Inform the "destination" that swapping has failed.
            su_msg: SwapUpdateMsg = {
                "cmd": "SWAP_UPDATE",
                "path_id": msg["path_id"],
                "swapping_node": msg["swapping_node"],
                "partner": msg["partner"],
                "epr": my_new_epr.name,
                "new_epr": None,
            }
            self.fw.send_msg(destination, su_msg, fib_entry)
            return

        # The swapping_node successfully swapped in parallel with this node.
        # Determine the "destination" and "partner".
        # Merge the two swaps (physically already happened).
        new_epr.read = True
        if other_epr.dst == self.node:  # destination is to the left of own node
            merged_epr = Entanglement.swap(other_epr, new_epr, now=self.simulator.tc)
            partner = cast(QNode, new_epr.dst)
            destination = cast(QNode, other_epr.src)
        else:  # destination is to the right of own node
            merged_epr = Entanglement.swap(new_epr, other_epr, now=self.simulator.tc)
            partner = cast(QNode, new_epr.src)
            destination = cast(QNode, other_epr.dst)
        assert partner.name == msg["partner"]

        if merged_epr is not None:
            self.fw.cnt.n_swapped_p += 1

            # adjust EPR paths for dynamic EPR affectation and statistical mux
            self.mux.su_parallel_succeeded(merged_epr, new_epr, other_epr)

            # Inform the "destination" of the swap result and new "partner".
            self._deposit_remote_swapped_epr(destination, merged_epr)

        su_msg: SwapUpdateMsg = {
            "cmd": "SWAP_UPDATE",
            "path_id": msg["path_id"],
            "swapping_node": msg["swapping_node"],
            "partner": partner.name,
            "epr": my_new_epr.name,
            "new_epr": None if merged_epr is None else merged_epr.name,
        }
        self.fw.send_msg(destination, su_msg, fib_entry)

        # Update records to support potential parallel swapping with "partner".
        _, p_rank = fib_entry.find_index_and_swap_rank(partner.name)
        if fib_entry.own_swap_rank == p_rank and merged_epr is not None:
            self.parallel_swappings[new_epr.name] = (new_epr, other_epr, merged_epr)
