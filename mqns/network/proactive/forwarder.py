#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

from typing_extensions import override

from mqns.entity.cchannel import ClassicPacket, RecvClassicPacket
from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import Application, Node, QNode
from mqns.models.epr import WernerStateEntanglement
from mqns.network.network import TimingPhase, TimingPhaseEvent
from mqns.network.proactive.fib import Fib, FibEntry
from mqns.network.proactive.message import InstallPathMsg, PurifResponseMsg, PurifSolicitMsg, SwapUpdateMsg, UninstallPathMsg
from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.mux_buffer_space import MuxSchemeBufferSpace
from mqns.network.proactive.select import (
    SelectPurifQubit,
    SelectSwapQubit,
    select_purif_qubit,
)
from mqns.network.protocol.event import ManageActiveChannels, QubitEntangledEvent, QubitReleasedEvent
from mqns.simulator import Simulator
from mqns.utils import log


class ProactiveForwarderCounters:
    def __init__(self):
        self.n_entg = 0
        """how many elementary entanglements received from link layer"""
        self.n_purif: list[int] = []
        """how many entanglements completed i-th purif round (zero-based index)"""
        self.n_eligible = 0
        """how many entanglements completed all purif rounds and became eligible"""
        self.n_swapped_s = 0
        """how many swaps succeeded sequentially"""
        self.n_swapped_p = 0
        """how many swaps succeeded with parallel merging"""
        self.n_swap_conflict = 0
        """how many swaps were skipped due to conflictual decisions"""
        self.n_consumed = 0
        """how many entanglements were consumed (either end-to-end or in swap-disabled mode)"""
        self.consumed_sum_fidelity = 0.0
        """sum of fidelity of consumed entanglements"""

    def increment_n_purif(self, i: int):
        if len(self.n_purif) <= i:
            self.n_purif += [0] * (i + 1 - len(self.n_purif))
        self.n_purif[i] += 1

    @property
    def n_swapped(self) -> int:
        """how many swaps succeeded"""
        return self.n_swapped_s + self.n_swapped_p

    @property
    def consumed_avg_fidelity(self) -> float:
        """average fidelity of consumed entanglements"""
        return 0.0 if self.n_consumed == 0 else self.consumed_sum_fidelity / self.n_consumed

    def __repr__(self) -> str:
        return (
            f"entg={self.n_entg} purif={self.n_purif} eligible={self.n_eligible} "
            f"swapped={self.n_swapped_s}+{self.n_swapped_p} "
            f"swap-conflict={self.n_swap_conflict} "
            f"consumed={self.n_consumed} (F={self.consumed_avg_fidelity})"
        )


class ProactiveForwarder(Application):
    """
    ProactiveForwarder is the forwarder of QNodes and receives routing instructions from the controller.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    def __init__(
        self,
        *,
        ps: float = 1.0,
        mux: MuxScheme = MuxSchemeBufferSpace(),
        select_purif_qubit: SelectPurifQubit = None,
        select_swap_qubit: SelectSwapQubit = None,
    ):
        """
        This constructor sets up a node's entanglement forwarding logic in a quantum network.
        It configures the swapping success probability and preparing internal
        state for managing memory, routing instructions (via FIB), synchronization,
        and classical communication handling.

        Args:
            ps: Probability of successful entanglement swapping (default: 1.0).
            mux: Path multiplexing scheme (default: buffer-space).
            but serving the same S-D request.
        """
        super().__init__()

        assert 0.0 <= ps <= 1.0
        self.ps = ps
        """Probability of successful entanglement swapping."""
        self.mux = deepcopy(mux)
        """Multiplexing scheme."""
        self._select_purif_qubit = select_purif_qubit
        self._select_swap_qubit = select_swap_qubit

        self.fib = Fib()
        """FIB structure."""

        # event handlers
        self.add_handler(self.handle_sync_phase, TimingPhaseEvent)
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)
        self.add_handler(self.qubit_is_entangled, QubitEntangledEvent)

        self.waiting_etg: list[QubitEntangledEvent] = []
        """
        Elementary-entangled qubits received during EXTERNAL phase.
        These are buffered until INTERNAL phase starts.
        """

        self.waiting_su: dict[int, tuple[SwapUpdateMsg, FibEntry]] = {}
        """
        SwapUpdates received prior to QubitEntangledEvent.
        Key: MemoryQubit addr.
        Value: SwapUpdateMsg and FIBEntry.
        """

        self.parallel_swappings: dict[
            str, tuple[WernerStateEntanglement, WernerStateEntanglement, WernerStateEntanglement]
        ] = {}
        """
        Records for potential parallel swappings.
        See `_su_parallel` method.
        """

        self.remote_swapped_eprs: dict[str, WernerStateEntanglement] = {}
        """
        EPRs that have been swapped remotely but the SwapUpdateMsg have not arrived.
        Each key is an EPR name; each value is the EPR.

        When a remote forwarder performs a swapping in which this node is either src or dst of the new EPR,
        it deposits the swapped EPR here and transmits the corresponding SwapUpdateMsg.
        Upon receiving the SwapUpdateMsg, the local forwarder pops the EPR.

        XXX Current approach assumes cchannels do not have packet loss.
        """

        self.cnt = ProactiveForwarderCounters()
        """
        Counters.
        """

    @override
    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=QNode)
        self.memory = self.own.get_memory()
        self.net = self.own.network
        self.mux.fw = self

    def handle_sync_phase(self, event: TimingPhaseEvent):
        """
        Handle timing phase signals, only used in SYNC timing mode.

        Upon entering EXTERNAL phase:

        1. Clear `remote_swapped_eprs`.
           All memory qubits are being discarded by LinkLayer, so that these have become useless.

        Upon entering INTERNAL phase:

        1. Start processing elementary entanglements that arrived during EXTERNAL phase.
        """
        if event.phase == TimingPhase.EXTERNAL:
            self.remote_swapped_eprs.clear()
        elif event.phase == TimingPhase.INTERNAL:
            log.debug(f"{self.own}: there are {len(self.waiting_etg)} etg qubits to process")
            for etg_event in self.waiting_etg:
                self.qubit_is_entangled(etg_event)
            self.waiting_etg.clear()

    CLASSIC_CONTROL_HANDLERS: dict[str, Callable[["ProactiveForwarder", Any], None]] = {}
    CLASSIC_SIGNALING_HANDLERS: dict[str, Callable[["ProactiveForwarder", Any, FibEntry], None]] = {}

    def RecvClassicPacketHandler(self, event: RecvClassicPacket) -> bool:
        """
        Dispatch or forward a received classical packet.

        This method recognizes a message that is a dict with "cmd" key with known value.
        The message is then passed to the appropriate handler function.

        For a signaling message where own node is not the destination, it is forwarded along the path.

        Returns False for unrecognized message types, which allows the packet to go to the next application.
        """
        packet = event.packet
        msg = packet.get()
        if not (isinstance(msg, dict) and "cmd" in msg):
            return False
        cmd: str = msg["cmd"]
        handler = self.CLASSIC_CONTROL_HANDLERS.get(cmd)
        if handler is not None:
            log.debug(f"{self.own}: received control message from {packet.src} | {msg}")
            handler(self, msg)
            return True

        handler = self.CLASSIC_SIGNALING_HANDLERS.get(cmd)
        if handler is None:
            return False

        path_id: int = msg["path_id"]
        try:
            fib_entry = self.fib.get(path_id)
        except IndexError:
            log.debug(f"{self.own}: dropping signaling message from {packet.src}, reason=no-fib-entry | {msg}")
            return True

        if packet.dest != self.own:
            self.send_msg(packet.dest, msg, fib_entry, forward=True)
            return True

        log.debug(f"{self.own}: received signaling message from {packet.src} | {msg}")
        handler(self, msg, fib_entry)
        return True

    def send_msg(self, dest: Node, msg: Any, fib_entry: FibEntry, *, forward=False):
        """
        Send/forward a message along the path specified in FIB entry.
        """
        dest_idx = fib_entry.route.index(dest.name)
        nh = fib_entry.route[fib_entry.own_idx + 1] if dest_idx > fib_entry.own_idx else fib_entry.route[fib_entry.own_idx - 1]
        next_hop = self.own.network.get_node(nh)

        log.debug(
            f"{self.own}: {'forwarding' if forward else 'sending'} signaling message to {dest.name} via {next_hop.name} | {msg}"
        )
        self.own.get_cchannel(next_hop).send(ClassicPacket(msg, src=self.own, dest=dest), next_hop)

    def handle_install_path(self, msg: InstallPathMsg):
        """
        Process an install_path message containing routing instructions from the controller.

        1. Insert FIB entry.
        2. Identity neighbors and qchannels.
        3. Save the path and neighbors in the multiplexing scheme.
        4. Notify LinkLayer to start elementary EPR generation toward the right neighbor.
        """
        simulator = self.simulator
        path_id = msg["path_id"]
        instructions = msg["instructions"]
        self.mux.validate_path_instructions(instructions)

        # populate FIB
        route = instructions["route"]
        fib_entry = FibEntry(
            path_id=path_id,
            req_id=instructions["req_id"],
            route=route,
            own_idx=route.index(self.own.name),
            swap=instructions["swap"],
            purif=instructions["purif"],
        )
        self.fib.insert_or_replace(fib_entry)

        # identify left/right neighbors
        l_neighbor = self._find_neighbor(fib_entry, -1)
        r_neighbor = self._find_neighbor(fib_entry, +1)

        if l_neighbor:
            l_qchannel = self.own.get_qchannel(l_neighbor)

            # associate path with qchannel and allocate qubits
            self.mux.install_path_neighbor(instructions, fib_entry, PathDirection.LEFT, l_neighbor, l_qchannel)

        if r_neighbor:
            r_qchannel = self.own.get_qchannel(r_neighbor)

            # associate path with qchannel and allocate qubits
            self.mux.install_path_neighbor(instructions, fib_entry, PathDirection.RIGHT, r_neighbor, r_qchannel)

            # instruct LinkLayer to start generating EPRs on the qchannel toward the right neighbor
            simulator.add_event(
                ManageActiveChannels(
                    self.own,
                    r_neighbor,
                    r_qchannel,
                    path_id=path_id if self.mux.qubit_has_path_id() else None,
                    start=True,
                    t=simulator.tc,
                    by=self,
                )
            )

    CLASSIC_CONTROL_HANDLERS["install_path"] = handle_install_path

    def handle_uninstall_path(self, msg: UninstallPathMsg):
        """
        Process an uninstall_path message containing routing instructions from the controller.

        1. Insert FIB entry.
        2. Identity neighbors and qchannels.
        3. Save the path and neighbors in the multiplexing scheme.
        4. Notify LinkLayer to start elementary EPR generation toward the right neighbor.
        """
        simulator = self.simulator
        path_id = msg["path_id"]

        # retrieve and erase FIB entry
        fib_entry = self.fib.get(path_id)
        self.fib.erase(path_id)

        # identify left/right neighbors
        l_neighbor = self._find_neighbor(fib_entry, -1)
        r_neighbor = self._find_neighbor(fib_entry, +1)

        if l_neighbor:
            l_qchannel = self.own.get_qchannel(l_neighbor)

            # disassociate path with qchannel and deallocate qubits
            _ = l_qchannel
            self.mux.uninstall_path_neighbor(fib_entry, PathDirection.LEFT, l_neighbor, l_qchannel)

        if r_neighbor:
            r_qchannel = self.own.get_qchannel(r_neighbor)

            # disassociate path with qchannel and deallocate qubits
            self.mux.uninstall_path_neighbor(fib_entry, PathDirection.RIGHT, r_neighbor, r_qchannel)

            # instruct LinkLayer to stop generating EPRs on the qchannel toward the right neighbor
            simulator.add_event(
                ManageActiveChannels(
                    self.own,
                    r_neighbor,
                    r_qchannel,
                    path_id=path_id if self.mux.qubit_has_path_id() else None,
                    start=False,
                    t=simulator.tc,
                    by=self,
                )
            )

    CLASSIC_CONTROL_HANDLERS["uninstall_path"] = handle_uninstall_path

    def _find_neighbor(self, fib_entry: FibEntry, route_offset: int) -> QNode | None:
        neigh_idx = fib_entry.own_idx + route_offset
        if neigh_idx in (-1, len(fib_entry.route)):  # no left/right neighbor if own node is the left/right end node
            return None
        return self.net.get_node(fib_entry.route[neigh_idx])

    def qubit_is_entangled(self, event: QubitEntangledEvent):
        """
        Handle a qubit entering ENTANGLED state, i.e. having an elementary entanglement.

        In ASYNC timing mode, events are processed immediately.
        In SYNC timing mode, events arrive in EXTERNAL phase and is queued in `self.waiting_etg`.
        Queued events are released upon entering INTERNAL phase and then processed.

        The actual processing is handled by the multiplexing scheme.

        If a SwapUpdate was received before processing this event and buffered in `self.waiting_su`,
        it is re-processed at this time.

        Args:
            event: Event containing the entangled qubit and its associated metadata (e.g., neighbor).

        """
        if not self.own.timing.is_internal():  # in SYNC timing mode EXTERNAL phase
            self.waiting_etg.append(event)
            return

        self.cnt.n_entg += 1

        qubit = event.qubit
        assert qubit.state == QubitState.ENTANGLED1
        self.mux.qubit_is_entangled(qubit, event.neighbor)

        su_args = self.waiting_su.pop(qubit.addr, None)
        if (
            qubit.state != QubitState.RELEASE  # qubit was released due to uninstalled path
            and su_args
        ):
            self.handle_swap_update(*su_args)

    def qubit_is_purif(self, qubit: MemoryQubit, fib_entry: FibEntry, partner: QNode):
        """
        Handle a qubit entering PURIF state or have completed a previous purification round.

        1. Determines the segment in which the qubit is entangled and number of required purification rounds.
        2. If the required rounds are completed, the qubit becomes eligible.
        3. Otherwise, check if own node is primary for the purification protocol.
           If so, search for an auxiliary qubit to use, release the auxiliary qubit,
           and send PURIF_SOLICIT to the partner node.

        Args:
            qubit: The memory qubit at PURIF state.
            fib_entry: FIB entry containing routing and purification instructions.
            partner: The node with which the qubit shares an EPR.
        """
        assert qubit.state == QubitState.PURIF
        assert qubit.qchannel is not None

        own_idx, own_rank = fib_entry.own_idx, fib_entry.own_swap_rank
        partner_idx, partner_rank = fib_entry.find_index_and_swap_rank(partner.name)
        if own_rank > partner_rank:
            # swapping order disallows initiating purif / swap / consumption
            return

        segment_name = f"{self.own.name}-{partner.name}" if own_idx < partner_idx else f"{partner.name}-{self.own.name}"
        want_rounds = fib_entry.purif.get(segment_name, 0)
        log.debug(
            f"{self.own}: segment {segment_name} (qubit {qubit.addr}) has "
            + f"{qubit.purif_rounds} and needs {want_rounds} purif rounds"
        )

        if qubit.purif_rounds == want_rounds:
            self.cnt.n_eligible += 1
            qubit.purif_rounds = 0
            qubit.state = QubitState.ELIGIBLE
            self.qubit_is_eligible(qubit, fib_entry)
            return
        assert qubit.purif_rounds < want_rounds

        is_primary = (own_rank, own_idx) < (partner_rank, partner_idx)
        if not is_primary:
            log.debug(f"{self.own}: is not primary node for segment {segment_name} purif")
            return

        found = select_purif_qubit(
            self._select_purif_qubit,
            qubit,
            fib_entry,
            partner,
            self.memory.find(
                lambda q, v: q.addr != qubit.addr  # not the same qubit
                and q.state == QubitState.PURIF  # in PURIF state
                and q.purif_rounds == qubit.purif_rounds  # with same number of purif rounds
                and partner in (v.src, v.dst)  # with the same partner
                and q.path_id == fib_entry.path_id,  # on the same path_id
                has_epr=True,
            ),
        )
        if not found:
            log.debug(f"{self.own}: no candidate EPR for segment {segment_name} purif round {1 + qubit.purif_rounds}")
            return

        self._send_purif_solicit(qubit, found[0], fib_entry, partner)

    def _send_purif_solicit(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry, partner: QNode):
        """
        Initiate purification protocol.

        Args:
            mq0: first memory qubit, which would be kept if purification succeeds.
            mq1: second memory qubit, which is consumed during purification.
            fib_entry: FIB entry.
            partner: quantum node with which entanglements are shared.
        """
        # read qubits to set fidelity at this time
        _, epr0 = self.memory.read(mq0.addr, destructive=False, must=True)
        _, epr1 = self.memory.read(mq1.addr, must=True)
        assert isinstance(epr0, WernerStateEntanglement)
        assert isinstance(epr1, WernerStateEntanglement)

        log.debug(
            f"{self.own}: request purif qubit {mq0.addr} (F={epr0.fidelity}) and "
            + f"{mq1.addr} (F={epr1.fidelity}) with partner {partner.name}"
        )

        mq0.state = QubitState.PENDING
        self.release_qubit(mq1)

        # send purif_solicit to partner
        msg: PurifSolicitMsg = {
            "cmd": "PURIF_SOLICIT",
            "path_id": fib_entry.path_id,
            "purif_node": self.own.name,
            "partner": partner.name,
            "epr": epr0.name,
            "measure_epr": epr1.name,
            "round": mq0.purif_rounds,
        }
        self.send_msg(partner, msg, fib_entry)

    def handle_purif_solicit(self, msg: PurifSolicitMsg, fib_entry: FibEntry):
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
        mq0, epr0 = self.memory.read(msg["epr"], destructive=False, must=True)
        mq1, epr1 = self.memory.read(msg["measure_epr"], must=True)
        # TODO: handle the exception case when an EPR is decohered and not found in memory
        assert isinstance(epr0, WernerStateEntanglement)
        assert isinstance(epr1, WernerStateEntanglement)

        for mq in (mq0, mq1):
            assert mq.state == QubitState.PURIF
            assert mq.purif_rounds == msg["round"]

        assert msg["partner"] == self.own.name
        primary = self.own.network.get_node(msg["purif_node"])
        log.debug(
            f"{self.own}: perform purif qubit {mq0.addr} (F={epr0.fidelity}) and "
            + f"{mq1.addr} (F={epr1.fidelity}) for round {1 + mq0.purif_rounds} with primary {primary.name}"
        )

        # perform purification between EPRs
        result = epr0.purify(epr1)
        log.debug(
            f"{self.own}: purif {'succeeded' if result else 'failed'} on qubit {mq0.addr} (F={epr0.fidelity}) "
            + f"for round {1 + mq0.purif_rounds} with primary {primary.name}"
        )

        if result:
            self.memory.update(old_qm=epr0.name, new_qm=epr0)
            self.cnt.increment_n_purif(mq0.purif_rounds)
            mq0.purif_rounds += 1
            mq0.state = QubitState.PURIF
            self.qubit_is_purif(mq0, fib_entry, primary)
        else:
            # in case of purification failure, release mq0
            self.release_qubit(mq0, read=True)

        # release mq1; destructive reading is already performed
        self.release_qubit(mq1)

        # send response message
        resp: PurifResponseMsg = {
            **msg,
            "cmd": "PURIF_RESPONSE",
            "result": result,
        }
        self.send_msg(primary, resp, fib_entry)

    CLASSIC_SIGNALING_HANDLERS["PURIF_SOLICIT"] = handle_purif_solicit

    def handle_purif_response(self, msg: PurifResponseMsg, fib_entry: FibEntry):
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
        qubit, epr = self.memory.get(msg["epr"], must=True)
        # TODO: handle the exception case when an EPR is decohered and not found in memory
        assert isinstance(epr, WernerStateEntanglement)

        result = msg["result"]
        log.debug(
            f"{self.own}: purif {'succeeded' if result else 'failed'} on qubit {qubit.addr} (F={epr.fidelity}) "
            + f"for round {1 + qubit.purif_rounds} with partner {msg['partner']}"
        )

        if not result:  # purif failed
            self.release_qubit(qubit, read=True)
            return

        # purif succeeded
        self.memory.update(old_qm=epr.name, new_qm=epr)
        self.cnt.increment_n_purif(qubit.purif_rounds)
        qubit.purif_rounds += 1
        qubit.state = QubitState.PURIF
        self.qubit_is_purif(qubit, fib_entry, self.own.network.get_node(msg["partner"]))

    CLASSIC_SIGNALING_HANDLERS["PURIF_RESPONSE"] = handle_purif_response

    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FibEntry | None):
        """
        Handle a qubit entering ELIGIBLE state.

        If this is an end node of the path, consume the EPR.

        Otherwise, attempt entanglement swapping:

        1. Look for a matching eligible qubit to perform swapping.
        2. Generate a new EPR if successful.
        3. Notify adjacent nodes with SWAP_UPDATE messages.

        Args:
            qubit: The qubit that became eligible.
            fib_entry: FIB entry (not available with MuxSchemeStatistical).
        """
        assert qubit.state == QubitState.ELIGIBLE
        if not self.own.timing.is_internal():
            log.debug(f"{self.own}: INT phase is over -> stop swaps")
            return

        _, epr = self.memory.get(qubit.addr, must=True)
        assert isinstance(epr, WernerStateEntanglement)
        if self.can_consume(fib_entry, epr):
            self.consume_and_release(qubit)
            return

        swap_candidate_tuple = self.mux.find_swap_candidate(qubit, epr, fib_entry)
        if swap_candidate_tuple:
            mq1, fib_entry = swap_candidate_tuple
            self.do_swapping(qubit, mq1, fib_entry)

    def do_swapping(
        self,
        mq0: MemoryQubit,
        mq1: MemoryQubit,
        fib_entry: FibEntry,
    ):
        """
        Perform swapping between two qubits at an intermediate node.
        These qubits must be in ELIGIBLE state and come from different qchannels.
        Partners are notified with SWAP_UPDATE messages.
        """
        assert mq0.addr != mq1.addr
        assert mq0.state == QubitState.ELIGIBLE
        assert mq1.state == QubitState.ELIGIBLE

        # Read both qubits and remove them from memory.
        #
        # One of these qubits must be entangled with a partner node to the left of the current node.
        # This is determined by epr.dst==self.own condition, because LinkLayer establishes elementary
        # entanglements from left to right, and swapping maintains this condition.
        # This qubit and related objects are assigned to prev_* variables.
        #
        # Likewise, the other qubit entangled with a partner node to the right is assigned to next_*.
        prev_tuple: tuple[QNode, MemoryQubit, WernerStateEntanglement] | None = None
        next_tuple: tuple[QNode, MemoryQubit, WernerStateEntanglement] | None = None
        for addr in (mq0.addr, mq1.addr):
            qubit, epr = self.memory.read(addr, must=True)
            assert isinstance(epr, WernerStateEntanglement)
            if epr.dst == self.own:
                assert epr.src is not None
                prev_tuple = epr.src, qubit, epr
            elif epr.src == self.own:
                assert epr.dst is not None
                next_tuple = epr.dst, qubit, epr
            else:
                raise Exception(f"Unexpected: swapping EPRs {mq0} x {mq1}")

        # Make sure both partners are found.
        assert prev_tuple is not None
        assert next_tuple is not None
        prev_partner, prev_qubit, prev_epr = prev_tuple
        next_partner, next_qubit, next_epr = next_tuple

        # Save ch_index metadata field onto elementary EPR.
        if not prev_epr.orig_eprs:
            prev_epr.ch_index = fib_entry.own_idx - 1
        if not next_epr.orig_eprs:
            next_epr.ch_index = fib_entry.own_idx

        # Attempt the swap.
        new_epr = prev_epr.swapping(epr=next_epr, ps=self.ps)
        log.debug(f"{self.own}: SWAP {'SUCC' if new_epr else 'FAILED'} | {prev_qubit} x {next_qubit}")

        if new_epr is not None:  # swapping succeeded
            self.cnt.n_swapped_s += 1

            # Update properties in newly generated EPR.
            new_epr.src = prev_partner
            new_epr.dst = next_partner

            # Inform multiplexing scheme.
            self.mux.swapping_succeeded(prev_epr, next_epr, new_epr)

        for (a_partner, a_qubit, a_epr), (b_partner, _, b_epr) in ((prev_tuple, next_tuple), (next_tuple, prev_tuple)):
            if new_epr is not None:
                # Keep records to support potential parallel swapping.
                _, a_rank = fib_entry.find_index_and_swap_rank(a_partner.name)
                if fib_entry.own_swap_rank == a_rank:
                    self.parallel_swappings[a_epr.name] = (a_epr, b_epr, new_epr)

                # Deposit swapped EPR at the partner.
                a_partner.get_app(ProactiveForwarder).remote_swapped_eprs[new_epr.name] = new_epr

            # Send SWAP_UPDATE to the partner.
            su_msg: SwapUpdateMsg = {
                "cmd": "SWAP_UPDATE",
                "path_id": fib_entry.path_id,
                "swapping_node": self.own.name,
                "partner": b_partner.name,
                "epr": a_epr.name,
                "new_epr": None if new_epr is None else new_epr.name,
            }
            self.send_msg(a_partner, su_msg, fib_entry)

            # Release old qubit.
            self.release_qubit(a_qubit)

    def handle_swap_update(self, msg: SwapUpdateMsg, fib_entry: FibEntry):
        """
        Process an SWAP_UPDATE signaling message.
        It may either update local qubit state or release decohered pairs.

        If QubitEntangledEvent for the qubit has not been processed, the SwapUpdate is buffered
        in `self.waiting_su` and will be re-tried after processing the QubitEntangledEvent.

        Args:
            msg: The SWAP_UPDATE message.
            fib_entry: FIB entry associated with path_id in the message.

        """
        if not self.own.timing.is_internal():
            log.debug(f"{self.own}: INT phase is over -> stop swaps")
            return

        _, sender_rank = fib_entry.find_index_and_swap_rank(msg["swapping_node"])
        if fib_entry.own_swap_rank < sender_rank:
            log.debug(f"### {self.own}: VERIFY -> rcvd SU from higher-rank node")
            return

        new_epr_name = msg["new_epr"]
        new_epr = None if new_epr_name is None else self.remote_swapped_eprs.pop(new_epr_name)

        epr_name = msg["epr"]
        qubit_pair = self.memory.get(epr_name)
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
            log.debug(f"### {self.own}: EPR {epr_name} decohered during SU transmissions")

    CLASSIC_SIGNALING_HANDLERS["SWAP_UPDATE"] = handle_swap_update

    def _su_sequential(
        self,
        msg: SwapUpdateMsg,
        fib_entry: FibEntry,
        qubit: MemoryQubit,
        new_epr: WernerStateEntanglement | None,
        maybe_purif: bool,
    ):
        """
        Process SWAP_UPDATE message where the local MemoryQubit still exists.
        This means the swapping was performed sequentially and local MemoryQubit has not decohered.

        Args:
            maybe_purif: whether the new EPR may enter PURIF state.
                         Set to True if own rank is higher than sender rank.
        """
        simulator = self.simulator
        if (
            new_epr is None  # swapping failed
            or (new_epr.decoherence_time is not None and new_epr.decoherence_time <= simulator.tc)  # oldest pair decohered
        ):
            if new_epr:
                log.debug(f"{self.own}: NEW EPR {new_epr} decohered during SU transmissions")
            # Inform LinkLayer that the memory qubit has been released.
            self.release_qubit(qubit, read=True)
            return

        # Update old EPR with new EPR (fidelity and partner).
        updated = self.memory.update(old_qm=msg["epr"], new_qm=new_epr)
        if not updated:
            raise Exception(f"{self.own}: EPR update failed | old={msg['epr']} , new={new_epr}")

        if maybe_purif:
            # If own rank is higher than sender rank but lower than new partner rank,
            # it is our turn to purify the qubit and progress toward swapping.
            qubit.purif_rounds = 0
            qubit.state = QubitState.PURIF
            partner = self.own.network.get_node(msg["partner"])
            self.qubit_is_purif(qubit, fib_entry, partner)

    def _su_parallel(self, msg: SwapUpdateMsg, fib_entry: FibEntry, new_epr: WernerStateEntanglement | None):
        """
        Process SWAP_UPDATE message during parallel swapping.
        """
        simulator = self.simulator
        (shared_epr, other_epr, my_new_epr) = self.parallel_swappings.pop(msg["epr"])
        _ = shared_epr

        # safety in statistical mux to avoid conflictual swappings on different paths
        if self.mux.su_parallel_has_conflict(my_new_epr, msg["path_id"]):
            self.cnt.n_swap_conflict += 1
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
            or (new_epr.decoherence_time is not None and new_epr.decoherence_time <= simulator.tc)  # oldest pair decohered
        ):
            # Determine the "destination".
            if other_epr.dst == self.own:  # destination is to the left of own node
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
            self.send_msg(destination, su_msg, fib_entry)
            return

        # The swapping_node successfully swapped in parallel with this node.
        # Merge the two swaps (physically already happened).
        merged_epr = new_epr.swapping(epr=other_epr)

        # Determine the "destination" and "partner".
        if other_epr.dst == self.own:  # destination is to the left of own node
            if merged_epr is not None:
                merged_epr.src = other_epr.src
                merged_epr.dst = new_epr.dst
            partner = new_epr.dst
            destination = other_epr.src
        else:  # destination is to the right of own node
            if merged_epr is not None:
                merged_epr.src = new_epr.src
                merged_epr.dst = other_epr.dst
            partner = new_epr.src
            destination = other_epr.dst
        assert partner is not None
        assert partner.name == msg["partner"]
        assert destination is not None

        if merged_epr is not None:
            self.cnt.n_swapped_p += 1

        # adjust EPR paths for dynamic EPR affectation and statistical mux
        if merged_epr is not None:
            self.mux.su_parallel_succeeded(merged_epr, new_epr, other_epr)

        # Inform the "destination" of the swap result and new "partner".
        if merged_epr is not None:
            destination.get_app(ProactiveForwarder).remote_swapped_eprs[merged_epr.name] = merged_epr

        su_msg: SwapUpdateMsg = {
            "cmd": "SWAP_UPDATE",
            "path_id": msg["path_id"],
            "swapping_node": msg["swapping_node"],
            "partner": partner.name,
            "epr": my_new_epr.name,
            "new_epr": None if merged_epr is None else merged_epr.name,
        }
        self.send_msg(destination, su_msg, fib_entry)

        # Update records to support potential parallel swapping with "partner".
        _, p_rank = fib_entry.find_index_and_swap_rank(partner.name)
        if fib_entry.own_swap_rank == p_rank and merged_epr is not None:
            self.parallel_swappings[new_epr.name] = (new_epr, other_epr, merged_epr)

    def can_consume(self, fib_entry: FibEntry | None, epr: WernerStateEntanglement) -> bool:
        if fib_entry is None:
            assert epr.src is not None
            assert epr.dst is not None
            src, dst = epr.src.name, epr.dst.name
            return next(self.fib.find_request(lambda g: g.src == src and g.dst == dst), None) is not None

        return fib_entry.is_swap_disabled or fib_entry.own_idx in (0, len(fib_entry.route) - 1)

    def consume_and_release(self, qubit: MemoryQubit):
        """
        Consume an entangled qubit.
        """
        _, qm = self.memory.read(qubit.addr, must=True)
        assert isinstance(qm, WernerStateEntanglement)
        assert qm.src is not None
        assert qm.dst is not None

        log.debug(f"{self.own}: consume EPR: {qm.name} -> {qm.src.name}-{qm.dst.name} | F={qm.fidelity}")
        self.cnt.n_consumed += 1
        self.cnt.consumed_sum_fidelity += qm.fidelity

        self.release_qubit(qubit)

    def release_qubit(self, qubit: MemoryQubit, *, read=False):
        """
        Release a qubit.

        Args:
            read: whether to perform a destructive read.
        """
        simulator = self.simulator

        if read:
            self.memory.read(qubit.addr)

        qubit.state = QubitState.RELEASE
        simulator.add_event(QubitReleasedEvent(self.own, qubit, t=simulator.tc, by=self))
