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

import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TypedDict, cast

from qns.entity.cchannel import ClassicPacket, RecvClassicPacket
from qns.entity.memory import QuantumMemory
from qns.entity.memory.memory_qubit import MemoryQubit, PathDirection, QubitState
from qns.entity.node import Application, Node, QNode
from qns.models.core import QuantumModel
from qns.models.epr import WernerStateEntanglement
from qns.network import QuantumNetwork, SignalTypeEnum, TimingModeEnum
from qns.network.protocol.event import ManageActiveChannels, QubitEntangledEvent, QubitReleasedEvent, TypeEnum
from qns.network.protocol.fib import FIBEntry, ForwardingInformationBase, find_index_and_swapping_rank, is_isolated_links
from qns.simulator import Simulator
from qns.utils import log

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


MultiplexingVector = list[tuple[int, int]]


class InstallPathInstructions(TypedDict):
    req_id: int
    route: list[str]
    swap: list[int]
    mux: Literal["B", "S"]
    m_v: NotRequired[MultiplexingVector]
    purif: dict[str, int]


class InstallPathMsg(TypedDict):
    cmd: Literal["install_path"]
    path_id: int
    instructions: InstallPathInstructions


class PurifMsgBase(TypedDict):
    path_id: int
    purif_node: str
    partner: str
    epr: str
    measure_epr: str
    round: int


class PurifSolicitMsg(PurifMsgBase):
    cmd: Literal["PURIF_SOLICIT"]


class PurifResponseMsg(PurifMsgBase):
    cmd: Literal["PURIF_RESPONSE"]
    result: bool


class SwapUpdateMsg(TypedDict):
    cmd: Literal["SWAP_UPDATE"]
    path_id: int
    swapping_node: str
    partner: str
    epr: str
    new_epr: str | None  # None means swapping failed


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
        self.n_consumed = 0
        """how many entanglements were consumed (either end-to-end or in isolated links mode)"""
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
            + f"swapped={self.n_swapped_s}+{self.n_swapped_p} "
            + f"consumed={self.n_consumed} (F={self.consumed_avg_fidelity})"
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
        isolate_paths: bool = True,
        statistical_mux: bool = False,
        path_select_fn: Callable[[list[FIBEntry]], int] | None = None,
    ):
        """This constructor sets up a node's entanglement forwarding logic in a quantum network.
        It configures the swapping success probability and preparing internal
        state for managing memory, routing instructions (via FIB), synchronization,
        and classical communication handling.

        Parameters
        ----------
            ps (float): Probability of successful entanglement swapping (default: 1.0).
            isolate_paths (bool): Whether to allow the swapping of qubits allocated to different paths
            but serving the same S-D request.
            statistical_mux (bool): When qubit-path allocation is disabled, use statistical multiplexing
            or the default dynamic random EPR affectation.
            path_select_fn: custom path selection function for dynamic EPR allocation
        """
        super().__init__()

        self.ps = ps
        """Probability of successful entanglement swapping"""
        self.isolate_paths = isolate_paths
        """Whether to isolate or not paths serving the same request"""
        self.net: QuantumNetwork
        """quantum network instance"""
        self.own: QNode
        """quantum node this Forwarder equips"""
        self.memory: QuantumMemory
        """quantum memory of the node"""

        self.fib = ForwardingInformationBase()
        """FIB structure"""

        self.waiting_qubits: list[QubitEntangledEvent] = []
        """stores the qubits waiting for the INTERNAL phase (SYNC mode)"""

        self.request_paths_map = defaultdict[int, list[int]](lambda: [])
        """stores paths associated with the same S-D request (for non-isolated paths)"""

        self.qchannel_paths_map = defaultdict[str, list[int]](lambda: [])
        """stores path-qchannel relationship (for statistical mux)"""

        self.path_select_fn = path_select_fn or random_path_selector
        """stores path selection function for dynamic EPR allocation"""

        # event handlers
        self.add_handler(self.RecvClassicPacketHandler, RecvClassicPacket)
        self.add_handler(self.qubit_is_entangled, QubitEntangledEvent)

        self.parallel_swappings: dict[
            str, tuple[WernerStateEntanglement, WernerStateEntanglement, WernerStateEntanglement]
        ] = {}
        """manage potential parallel swappings"""

        self.remote_swapped_eprs: dict[str, WernerStateEntanglement] = {}
        """
        EPRs that have been swapped remotely but the SwapUpdateMsg have not arrived.
        Each key is an EPR name; each value is the EPR.

        When a remote forwarder performs a swapping in which this node is either src or dst of the new EPR,
        it deposits the swapped EPR here and transmits the corresponding SwapUpdateMsg.
        Upon receiving the SwapUpdateMsg, the local forwarder pops the EPR.

        XXX Current approach assumes cchannels do not have packet loss.
        """

        self.statistical_mux = statistical_mux
        """to enable statistical mux"""

        self.cnt = ProactiveForwarderCounters()

    def install(self, node: Node, simulator: Simulator):
        """called at initialization of the node"""
        super().install(node, simulator)
        self.own = self.get_node(node_type=QNode)
        self.memory = self.own.get_memory()
        self.net = self.own.network

    CLASSIC_SIGNALING_HANDLERS: dict[str, Callable[["ProactiveForwarder", Any, FIBEntry], None]] = {}

    def RecvClassicPacketHandler(self, event: RecvClassicPacket) -> bool:
        """
        Receives a classical packet and dispatches it as control or signaling.

        If the message is originated from a Controller, it is treated as a control message and passed to `handle_control`.

        If the message is originated from another node type and contains a `cmd` field recognized as a signaling message:
        - If the current node is the destination, it is dispatched to the corresponding signaling command handler.
        - If the current node is not the destination, it is forwarded along the path.

        Returns False for unrecognized message types, which allows the classic packet to go to the next application.

        """
        packet = event.packet
        msg = packet.get()
        if not (isinstance(msg, dict) and "cmd" in msg):
            return False
        if msg["cmd"] == "install_path":
            return self.handle_install_path(cast(InstallPathMsg, msg))
        if msg["cmd"] not in self.CLASSIC_SIGNALING_HANDLERS:
            return False

        log.debug(f"{self.own.name}: {msg}")

        path_id: int = msg["path_id"]
        fib_entry = self.fib.get_entry(path_id, must=True)

        assert packet.dest is not None
        if packet.dest.name != self.own.name:  # node is not destination: forward message
            self.send_msg(dest=packet.dest, msg=msg, route=fib_entry["path_vector"])
            return True

        self.CLASSIC_SIGNALING_HANDLERS[msg["cmd"]](self, msg, fib_entry)
        return True

    def handle_install_path(self, msg: InstallPathMsg) -> bool:
        """
        Processes an install_path message containing routing instructions from the controller.

        Determines left/right neighbors from the route, identifies corresponding quantum channels,
        and allocates qubits based on the multiplexing vector (for the buffer-space mode).
        Updates the FIB with path, swapping, and purification info, and triggers EPR generation via the
        link layer on the outgoing channel.
        No path allocation for qubits means statistical mux is required, but this is not implemented.
        """
        simulator = self.simulator
        path_id = msg["path_id"]
        instructions = msg["instructions"]
        request_id = instructions["req_id"]

        # maintain map of path ID associated with each request ID
        self.request_paths_map[request_id].append(path_id)

        route = instructions["route"]
        log.debug(f"{self.own.name}: routing instructions of path {path_id}: {instructions}")

        m_v = None
        if "m_v" in instructions:
            m_v = instructions["m_v"]
            assert len(m_v) + 1 == len(route)

        # identify left/right neighbors and allocate memory qubits to the path
        # example visualization:
        # route = [ S, R, D ]
        # m_v = [ (4,2) , (2,4) ]
        # S--(4,2)--R--(2,4)--D
        # here, R should allocate 2 qubits toward S.
        _, l_qubits = self._find_neighbor_and_allocate_qubits(
            path_id, route, route_offset=-1, m_v=m_v, m_v_offset=-1, ch_side=1, path_direction=PathDirection.LEFT
        )
        r_neighbor, r_qubits = self._find_neighbor_and_allocate_qubits(
            path_id, route, route_offset=1, m_v=m_v, m_v_offset=0, ch_side=0, path_direction=PathDirection.RIGHT
        )

        if m_v is not None:
            log.debug(f"{self.own}: Allocated qubits: left = {l_qubits} | right = {r_qubits}")

        # populate FIB
        self.fib.add_entry(
            replace=True,
            path_id=path_id,
            request_id=request_id,
            path_vector=route,
            swap_sequence=instructions["swap"],
            purification_scheme=instructions["purif"],
            qubit_addresses=[],
        )

        # instruct LinkLayer to start generating EPRs on the qchannel toward the right neighbor
        if r_neighbor:
            p = path_id if instructions["mux"] == "B" else None
            simulator.add_event(ManageActiveChannels(self.own, r_neighbor, TypeEnum.ADD, p, t=simulator.tc, by=self))

        # TODO: remove path, type=TypeEnum.REMOVE
        return True

    def _find_neighbor_and_allocate_qubits(
        self,
        path_id: int,
        route: list[str],
        *,
        route_offset: int,  # to locate which side of the route we want (left/right)
        m_v: MultiplexingVector | None,
        m_v_offset: int,  # to locate which qchannel we want in m_v
        ch_side: int,  # \in {0,1} to specify which side of the qchannel we want in m_v (i.e., inverse of the offset direction)
        path_direction: PathDirection,  # to void loops in non-isolated multipaths (see Memory)
    ) -> tuple[QNode | None, list[int] | None]:
        own_idx = route.index(self.own.name)
        neigh_idx = own_idx + route_offset
        if neigh_idx in (-1, len(route)):  # no left/right neighbor if own node is the left/right end node
            return None, []

        neighbor = self.net.get_node(route[neigh_idx])
        qchannel = self.own.get_qchannel(neighbor)
        self.qchannel_paths_map[qchannel.name].append(path_id)
        if m_v is None:
            return neighbor, None

        n_qubits = m_v[own_idx + m_v_offset][ch_side]
        if n_qubits == 0:  # 0 means use all qubits assigned to this qchannel
            n_qubits = len(self.memory.get_channel_qubits(qchannel.name))

        qubits = [
            self.memory.allocate(path_id=path_id, ch_name=qchannel.name, path_direction=path_direction) for _ in range(n_qubits)
        ]
        if -1 in qubits:
            raise RuntimeError(f"{self.own}: insufficient memory qubits toward {neighbor} for path {path_id}")
        return neighbor, qubits

    def qubit_is_entangled(self, event: QubitEntangledEvent):
        """
        Handle a qubit entering ENTANGLED state.
        QubitEntangledEvent is either delivered from simulator or dequeued from `self.waiting_qubits`.
        In SYNC timing mode, events are queued if the current phase is EXTERNAL.
        In ASYNC timing mode or INTERNAL sync phase, events are handled immediately.

        Newly arrived elementary entanglements are processed based on their path allocation.

        If the qubit is assigned a path (buffer-space multiplexing), the method checks its eligibility and,
        if conditions are met, transitions it to the PURIF state and calls `purif` method.

        If the qubit is unassigned (statistical multiplexing), it is currently ignored.

        Args:
            event: Event containing the entangled qubit and its associated metadata (e.g., neighbor).

        """
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase == SignalTypeEnum.EXTERNAL:
            # Accept new etg while we are in EXT phase
            # Assume t_coh > t_ext: QubitEntangledEvent events should correspond to different qubits, no redundancy
            self.waiting_qubits.append(event)
            return

        self.cnt.n_entg += 1

        qubit = event.qubit
        assert qubit.fsm.state == QubitState.ENTANGLED
        if qubit.path_id is not None:  # for buffer-space mux
            fib_entry = self.fib.get_entry(qubit.path_id, must=True)
            qubit.purif_rounds = 0
            qubit.fsm.to_purif()
            self.qubit_is_purif(qubit, fib_entry, event.neighbor)
        else:  # for dynamic affectation (~ controlled statistical)
            log.debug(f"{self.own}: Qubit not allocated to any path.")
            if qubit.qchannel is None:
                raise Exception(f"{self.own}: No qubit-qchannel assignment. Not supported.")
            if qubit.qchannel.name not in self.qchannel_paths_map:
                raise Exception(f"{self.own}: qchannel {qubit.qchannel.name} not mapped to any path.")

            # available for every elementary EPR:
            possible_path_ids = self.qchannel_paths_map[qubit.qchannel.name]

            # Statistical multiplexing
            if self.statistical_mux:
                log.debug(f"{self.own}: Statistical mux enabled")
                _, epr = self.memory.get(address=qubit.addr, must=True)
                assert isinstance(epr, WernerStateEntanglement)
                log.debug(f"{self.own}: qubit {qubit}, set possible path IDs = {possible_path_ids}")
                epr.tmp_path_ids = list(possible_path_ids)  # to coordinate decisions along the path
                if _can_enter_purif(self.own.name, event.neighbor.name):
                    self.own.get_qchannel(event.neighbor)  # ensure qchannel exists
                    qubit.fsm.to_purif()
                    self.qubit_is_purif(qubit)
                return

            # Dynamic EPR effectation (not statistical mux)
            # TODO: if paths have different swap policies
            #       -> consider only paths for which this qubit may be eligible ??
            # Default is to affect EPR to paths randomly
            log.debug(f"{self.own}: Dynamic EPR affectation enabled")
            _, epr = self.memory.get(address=qubit.addr, must=True)
            assert isinstance(epr, WernerStateEntanglement)
            if epr.tmp_path_ids is None:  # whatever neighbor is first
                fib_entries = [self.fib.get_entry(pid, must=True) for pid in possible_path_ids]
                path_id = self.path_select_fn(fib_entries)
                epr.tmp_path_ids = [path_id]

            fib_entry = self.fib.get_entry(epr.tmp_path_ids[0], must=True)
            self.own.get_qchannel(event.neighbor)  # ensure qchannel exists
            qubit.fsm.to_purif()
            self.qubit_is_purif(qubit, fib_entry, event.neighbor)

    def qubit_is_purif(self, qubit: MemoryQubit, fib_entry: FIBEntry | None = None, partner: QNode | None = None):
        """
        Handle a qubit entering PURIF state.
        Determines the segment in which the qubit is entangled and number of required purification rounds from the FIB.
        If the required rounds are completed, the qubit becomes eligible. Otherwise, the node evaluates
        whether it is the initiator for the purification (i.e., primary). If so, it searches for an auxiliary
        qubit to use, consumes the auxiliary, updates qubit states, and sends a PURIF_SOLICIT message
        to the partner node.

        Args:
            qubit: The memory qubit at PURIF state.
            fib_entry: FIB entry containing routing and purification instructions.
            partner: The node with which the qubit shares an EPR.
        """
        assert qubit.fsm.state == QubitState.PURIF
        assert qubit.qchannel is not None

        if fib_entry is None:  # temporarily ignoring purif scheme in statistical mux
            log.debug(f"{self.own}: no FIB associated to qubit -> set eligible")
            qubit.fsm.to_eligible()
            self.qubit_is_eligible(qubit, fib_entry)
            return

        assert partner is not None
        partner_idx, partner_rank = find_index_and_swapping_rank(fib_entry, partner.name)
        own_idx, own_rank = find_index_and_swapping_rank(fib_entry, self.own.name)
        if own_rank > partner_rank:
            # swapping order disallows initiating purif / swap / consumption
            return

        segment_name = f"{self.own.name}-{partner.name}" if own_idx < partner_idx else f"{partner.name}-{self.own.name}"
        want_rounds = fib_entry["purification_scheme"].get(segment_name, 0)
        log.debug(
            f"{self.own}: segment {segment_name} (qubit {qubit.addr}) has "
            + f"{qubit.purif_rounds} and needs {want_rounds} purif rounds"
        )

        if qubit.purif_rounds == want_rounds:
            self.cnt.n_eligible += 1
            qubit.purif_rounds = 0
            qubit.fsm.to_eligible()
            self.qubit_is_eligible(qubit, fib_entry)
            return
        assert qubit.purif_rounds < want_rounds

        is_primary = (own_rank, own_idx) < (partner_rank, partner_idx)
        if not is_primary:
            log.debug(f"{self.own}: is not primary node for segment {segment_name} purif")
            return

        candidate = self._select_purif_qubit(
            exc_address=qubit.addr,
            partner=partner.name,
            qchannel=qubit.qchannel.name,
            path_id=fib_entry["path_id"],
            purif_rounds=qubit.purif_rounds,
        )
        if not candidate:
            log.debug(f"{self.own}: no candidate EPR for segment {segment_name} purif round {1 + qubit.purif_rounds}")
            return

        self._send_purif_solicit(qubit, candidate, fib_entry, partner)

    def _select_purif_qubit(
        self, exc_address: int, partner: str, qchannel: str, path_id: int, purif_rounds: int
    ) -> MemoryQubit | None:
        """Searches for a candidate qubit in the PURIF state that is ready for purification
        with a given qubit. The candidate must be a different qubit with the same path ID,
        quantum channel, partner, and has undergone the same number of purification rounds
        (i.e., recurrent purification schedule).

        Parameters
        ----------
            exc_address (int): Memory address of the qubit to exclude from the search.
            partner (str): Name of the partner node entangled with the qubit.
            qchannel (str): Name of the quantum channel associated with the qubit.
            path_id (int): Identifier for the entanglement path to match.
            purif_rounds (int): Number of purification rounds completed.

        Returns
        -------
            Optional[MemoryQubit]: A compatible qubit for purification if found, otherwise None.

        """
        qubits = self.memory.search_purif_qubits(
            exc_address=exc_address, partner=partner, qchannel=qchannel, path_id=path_id, purif_rounds=purif_rounds
        )
        if qubits:
            return qubits[0][0]  # pick up one qubit
            # TODO: Other possible qubit selection criteria
        return None

    def _send_purif_solicit(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FIBEntry, partner: QNode):
        """
        Initiate purification protocol.

        Args:
            mq0: first memory qubit, which would be kept if purification succeeds.
            mq1: second memory qubit, which is consumed during purification.
            fib_entry: FIB entry.
            partner: quantum node with which entanglements are shared.
        """
        simulator = self.simulator

        # read qubits to set fidelity at this time
        _, epr0 = self.memory.read(address=mq0.addr, destructive=False, must=True)
        _, epr1 = self.memory.read(address=mq1.addr, must=True)
        assert isinstance(epr0, WernerStateEntanglement)
        assert isinstance(epr1, WernerStateEntanglement)

        log.debug(
            f"{self.own}: request purif qubit {mq0.addr} (F={epr0.fidelity}) and "
            + f"{mq1.addr} (F={epr1.fidelity}) with partner {partner.name}"
        )

        mq0.fsm.to_pending()
        mq1.fsm.to_release()
        simulator.add_event(QubitReleasedEvent(self.own, mq1, t=simulator.tc, by=self))

        # send purif_solicit to partner
        msg: PurifSolicitMsg = {
            "cmd": "PURIF_SOLICIT",
            "path_id": fib_entry["path_id"],
            "purif_node": self.own.name,
            "partner": partner.name,
            "epr": epr0.name,
            "measure_epr": epr1.name,
            "round": mq0.purif_rounds,
        }
        self.send_msg(dest=partner, msg=msg, route=fib_entry["path_vector"])

    def handle_purif_solicit(self, msg: PurifSolicitMsg, fib_entry: FIBEntry):
        """Processes a PURIF_SOLICIT message from a partner node as part of the purification protocol.
        Retrieves the target and auxiliary qubits from memory, verifies their states, and attempts
        purification. If successful, updates the EPR and sends a PURIF_RESPONSE with result=True;
        otherwise, marks both qubits for release and replies with result=False.

        Args:
            msg: Message containing purification parameters and EPR names.
            fib_entry: FIB entry associated with path_id in the message.

        """
        simulator = self.simulator

        # mq0 is the "kept" memory whose fidelity would be increased if purification succeeds
        # mq1 is the "measured" memory that is consumed during purification
        mq0, epr0 = self.memory.read(key=msg["epr"], destructive=False, must=True)
        mq1, epr1 = self.memory.read(key=msg["measure_epr"], must=True)
        # TODO: handle the exception case when an EPR is decohered and not found in memory
        assert isinstance(epr0, WernerStateEntanglement)
        assert isinstance(epr1, WernerStateEntanglement)

        for mq in (mq0, mq1):
            assert mq.fsm.state == QubitState.PURIF
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
            self.cnt.increment_n_purif(mq0.purif_rounds)
            mq0.purif_rounds += 1
            self.memory.update(old_qm=epr0.name, new_qm=epr0)
            mq0.fsm.to_purif()
            self.qubit_is_purif(mq0, fib_entry, primary)
        else:
            # in case of purification failure, release mq0
            self.memory.read(address=mq0.addr)  # destructive reading
            mq0.fsm.to_release()
            simulator.add_event(QubitReleasedEvent(self.own, mq0, t=simulator.tc, by=self))

        # release mq1; destructive reading is already performed
        mq1.fsm.to_release()
        simulator.add_event(QubitReleasedEvent(self.own, mq1, t=simulator.tc, by=self))

        # send response message
        resp: PurifResponseMsg = {
            **msg,
            "cmd": "PURIF_RESPONSE",
            "result": result,
        }
        self.send_msg(dest=primary, msg=resp, route=fib_entry["path_vector"])

    CLASSIC_SIGNALING_HANDLERS["PURIF_SOLICIT"] = handle_purif_solicit

    def handle_purif_response(self, msg: PurifResponseMsg, fib_entry: FIBEntry):
        """Handles a PURIF_RESPONSE message indicating the outcome of a purification attempt.
        If the current node is the destination and purification succeeded, the EPR is updated,
        the qubit's purification round counter is incremented, and the qubit may re-enter the
        purification process. If purification failed, the qubit is released.

        Parameters
        ----------
            msg: Response message containing the result and identifiers of the purified EPRs.
            fib_entry: FIB entry associated with path_id in the message.

        """
        simulator = self.simulator

        qubit, epr = self.memory.get(key=msg["epr"], must=True)
        # TODO: handle the exception case when an EPR is decohered and not found in memory
        assert isinstance(epr, WernerStateEntanglement)

        result = msg["result"]
        log.debug(
            f"{self.own}: purif {'succeeded' if result else 'failed'} on qubit {qubit.addr} (F={epr.fidelity}) "
            + f"for round {1 + qubit.purif_rounds} with partner {msg['partner']}"
        )

        if not result:  # purif failed
            self.memory.read(address=qubit.addr)  # destructive reading
            qubit.fsm.to_release()
            simulator.add_event(QubitReleasedEvent(self.own, qubit, t=simulator.tc, by=self))
            return

        # purif succeeded
        self.cnt.increment_n_purif(qubit.purif_rounds)
        qubit.purif_rounds += 1
        self.memory.update(old_qm=epr.name, new_qm=epr)
        qubit.fsm.to_purif()
        self.qubit_is_purif(qubit, fib_entry, self.own.network.get_node(msg["partner"]))

    CLASSIC_SIGNALING_HANDLERS["PURIF_RESPONSE"] = handle_purif_response

    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FIBEntry | None = None):
        """
        Handle a qubit entering ELIGIBLE state.

        If this is an end node of the path, consume the EPR.

        Otherwise, attempt entanglement swapping:
        1. Look for a matching eligible qubit to perform swapping.
        2. Generate a new EPR if successful.
        3. Notify adjacent nodes with SWAP_UPDATE messages.

        Parameters
        ----------
            qubit (MemoryQubit): The qubit that became eligible.
            fib_entry (dict, optional): FIB entry containing path and swap sequence.
            The only case where fib_entry is None is when statistical mux is used.
            In such case, the path ID (and thus FIB entry) is figured out from qubit/EPR
            eligibility at the time of calling the function.
        """
        assert qubit.fsm.state == QubitState.ELIGIBLE
        assert qubit.qchannel is not None
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.INTERNAL:
            log.debug(f"{self.own}: INT phase is over -> stop swaps")
            return

        if fib_entry is not None:
            if is_isolated_links(fib_entry):  # no swapping in isolated links
                self.consume_and_release(qubit)
                return

            route = fib_entry["path_vector"]
            own_idx = route.index(self.own.name)
            if own_idx in (0, len(route) - 1):  # this is an end node
                self.consume_and_release(qubit)
                return
        elif not self.own.name.startswith("R"):  # this is an end node
            self.consume_and_release(qubit)
            return

        # this is an intermediate node
        # look for another eligible qubit
        res = None
        epr: QuantumModel | None = None
        if fib_entry is not None:
            if qubit.path_id is not None:  # static qubit-path allocation is provided
                possible_path_ids = [fib_entry["path_id"]]
                if not self.isolate_paths:
                    # if not isolated paths -> include other paths serving the same request
                    possible_path_ids = self.request_paths_map[fib_entry["request_id"]]
                    log.debug(f"{self.own}: path ids {possible_path_ids}")

                res = self._select_eligible_qubit(
                    exc_qchannel=qubit.qchannel.name, exc_direction=qubit.path_direction, path_id=possible_path_ids
                )
            else:  # dynamic EPR-path allocation
                possible_path_ids = [fib_entry["path_id"]]
                res = self._select_eligible_qubit(exc_qchannel=qubit.qchannel.name, tmp_path_id=possible_path_ids)
        else:  # statistical mux
            # find qchannels whose qubits may be used with this qubit
            res = self.memory.get(address=qubit.addr)
            if not res:
                raise Exception(f"Cannot retrieve EPR for qubit {qubit}")
            _, epr = res
            assert isinstance(epr, WernerStateEntanglement)
            assert epr.tmp_path_ids is not None
            # use path_ids to look for acceptable qchannels for swapping, excluding the qubit's qchannel
            target_set = set(epr.tmp_path_ids)
            matched_channels = {
                channel
                for channel, path_ids in self.qchannel_paths_map.items()
                if target_set.intersection(path_ids) and channel != qubit.qchannel.name
            }
            # select qubits based on qchannels only
            res = self._select_eligible_qubit(
                exc_qchannel=qubit.qchannel.name, inc_qchannels=list(matched_channels), tmp_path_id=epr.tmp_path_ids
            )

        if res:  # do swapping
            path_ids = None
            if fib_entry is None:  # for statistical mux -> select a path
                res2 = self.memory.get(address=res.addr)
                if not res2:
                    raise Exception(f"Cannot retrieve EPR for other qubit {qubit}")
                _, epr2 = res2
                assert isinstance(epr, WernerStateEntanglement)
                assert epr.tmp_path_ids is not None
                assert isinstance(epr2, WernerStateEntanglement)
                assert epr2.tmp_path_ids is not None
                path_ids = select_common_element(epr.tmp_path_ids, epr2.tmp_path_ids)
                if not path_ids:
                    raise Exception(f"Cannot select path ID from {epr.tmp_path_ids} and {epr2.tmp_path_ids}")
                fib_entry = self.fib.get_entry(random.choice(path_ids), must=True)  # no need to be coordinated accross the path

            # Get both qubits
            # other_qubit, other_epr = self.memory.get(address=res.addr)
            # this_qubit, this_epr = self.memory.get(address=qubit.addr)

            other_fib_entry = fib_entry  # for all other cases except non-isolated paths

            # for non-isolated paths
            # if res.path_id is not None and qubit.path_id != res.path_id and not self.isolate_paths:
            #     log.debug(f"{self.own}: swapping in non-isolated paths {(qubit.path_id, res.path_id)}")
            #     endpoints = {this_epr.src.name, this_epr.dst.name, other_epr.src.name, other_epr.dst.name}

            #     route = fib_entry["path_vector"]
            #     fib_entry2 = self.fib.get_entry(res.path_id)
            #     route2 = fib_entry2["path_vector"]

            #     # Check if both partner nodes are in each route
            #     in_fib1 = endpoints.issubset(set(route))
            #     in_fib2 = endpoints.issubset(set(route2))

            #     if in_fib1 and not in_fib2:
            #         other_fib_entry = fib_entry
            #     elif in_fib2 and not in_fib1:
            #         fib_entry = fib_entry2
            #         other_fib_entry = fib_entry
            #     elif in_fib1 and in_fib2:
            #         other_fib_entry = fib_entry2
            #     else:
            #         raise Exception("Cannot find EPR endpoints in installed paths")

            self.do_swapping(qubit, res, fib_entry, other_fib_entry, path_ids)

    def _select_eligible_qubit(
        self,
        exc_qchannel: str,
        exc_direction: PathDirection | None = None,
        inc_qchannels: list[str] | None = None,
        path_id: list[int] | None = None,
        tmp_path_id: list[int] | None = None,
    ) -> MemoryQubit | None:
        """Searches for an eligible qubit in memory that matches the specified path ID and
        is located on a different qchannel than the excluded one. This is used to
        find a swap candidate during entanglement forwarding. Currently returns the first
        matching result found.

        Parameters
        ----------
            exc_qchannel (str): Name of the quantum channel to exclude from the search.
            exc_direction (PathDirection): Qubit direction to exclude to avoid loops.
            inc_qchannels (list[sr], optional): List of qchannel names the qubits should be assigned to.
            This is used with statistical mux when no qubit-path allocation is set.
            path_id (list[int], optional): List of identifiers for the paths to match against
            qubit.path_id in static qubit-path allocation.
            A list of path IDs is used to support multiple non-isolated paths serving the same request.
            tmp_path_id (list[int], optional): List of identifiers for the paths to match agains epr.tmp_path_ids
            in dynamic qubit allocation or statistical multiplexing.

        Returns
        -------
            Optional[MemoryQubit]: A single eligible memory qubit, if found; otherwise, None.

        """
        qubits = self.memory.search_eligible_qubits(
            exc_qchannel=exc_qchannel,
            exc_direction=exc_direction,
            inc_qchannels=inc_qchannels,
            path_id=path_id,
            tmp_path_id=tmp_path_id,
        )
        if qubits:
            # log.debug(f"{self.own}: eligible qubits: {qubits}")
            return qubits[0][0]  # pick up one qubit
            # TODO: Other qubit selection
            # (for statistical multiplexing, multipath, quasi-local swapping, etc.)
        return None

    def do_swapping(
        self,
        mq0: MemoryQubit,
        mq1: MemoryQubit,
        fib_entry: FIBEntry,
        other_fib_entry: FIBEntry,
        path_ids: list[int] | None = None,
    ):
        """
        Perform swapping between two qubits at an intermediate node.
        These qubits must be in ELIGIBLE state and come from different qchannels.
        Partners are notified with SWAP_UPDATE messages.
        """
        simulator = self.simulator

        # Read both qubits and remove them from memory.
        #
        # One of these qubits must be entangled with a partner node to the left of the current node.
        # This is determined by epr.dst==self.own condition, because LinkLayer establishes elementary
        # entanglements from left to right, and swapping maintains this condition.
        # This qubit and related objects are assigned to prev_* variables.
        #
        # Likewise, the other qubit entangled with a partner node to the right is assigned to next_*.
        prev_partner: QNode | None = None
        prev_qubit: MemoryQubit | None = None
        prev_epr: WernerStateEntanglement | None = None
        next_partner: QNode | None = None
        next_qubit: MemoryQubit | None = None
        next_epr: WernerStateEntanglement | None = None

        prev_fib_entry: FIBEntry | None = None
        next_fib_entry: FIBEntry | None = None

        for addr in (mq0.addr, mq1.addr):
            qubit, epr = self.memory.read(address=addr, must=True)
            assert isinstance(epr, WernerStateEntanglement)
            if epr.dst == self.own:
                prev_partner, prev_qubit, prev_epr = epr.src, qubit, epr
                prev_fib_entry = fib_entry if qubit.path_id == fib_entry["path_id"] else other_fib_entry
            elif epr.src == self.own:
                next_partner, next_qubit, next_epr = epr.dst, qubit, epr
                next_fib_entry = fib_entry if qubit.path_id == fib_entry["path_id"] else other_fib_entry
            else:
                raise Exception(f"Unexpected: swapping EPRs {mq0} x {mq1}")

        # Make sure both partners are found.
        assert prev_partner is not None
        assert prev_qubit is not None
        assert prev_epr is not None
        assert prev_fib_entry is not None
        assert next_partner is not None
        assert next_qubit is not None
        assert next_epr is not None
        assert next_fib_entry is not None

        prev_own_idx, prev_own_rank = find_index_and_swapping_rank(prev_fib_entry, self.own.name)
        next_own_idx, next_own_rank = find_index_and_swapping_rank(next_fib_entry, self.own.name)

        prev_route = prev_fib_entry["path_vector"]
        next_route = next_fib_entry["path_vector"]

        # Save ch_index metadata field onto elementary EPR.
        if not prev_epr.orig_eprs:
            prev_epr.ch_index = prev_own_idx - 1
        if not next_epr.orig_eprs:
            next_epr.ch_index = next_own_idx

        # Attempt the swap.
        new_epr = prev_epr.swapping(epr=next_epr, ps=self.ps)
        log.debug(f"{self.own}: SWAP {'SUCC' if new_epr else 'FAILED'} | {prev_qubit} x {next_qubit}")

        if new_epr is not None:  # swapping succeeded
            self.cnt.n_swapped_s += 1

            # Update properties in newly generated EPR.
            new_epr.src = prev_partner
            new_epr.dst = next_partner

            # for dynamic EPR affectation and statistical mux
            if prev_epr.tmp_path_ids is not None:
                if not self.statistical_mux:
                    assert prev_epr.tmp_path_ids == next_epr.tmp_path_ids
                    new_epr.tmp_path_ids = list(prev_epr.tmp_path_ids)
                else:
                    new_epr.tmp_path_ids = path_ids

            # another node just swapped on a shared EPR and changed its src/dst
            if prev_partner.name not in prev_route or next_partner.name not in next_route:
                raise Exception(f"{self.own}: Conflictual parallel swapping caused by SWAP-ASAP with non-isolated paths")

            # Keep records to support potential parallel swapping with prev_partner.
            _, prev_p_rank = find_index_and_swapping_rank(prev_fib_entry, prev_partner.name)
            if prev_own_rank == prev_p_rank:
                self.parallel_swappings[prev_epr.name] = (prev_epr, next_epr, new_epr)

            # Keep records to support potential parallel swapping with next_partner.
            _, next_p_rank = find_index_and_swapping_rank(next_fib_entry, next_partner.name)
            if next_own_rank == next_p_rank:
                self.parallel_swappings[next_epr.name] = (next_epr, prev_epr, new_epr)

        # Send SWAP_UPDATE to partners.
        for partner, old_epr, new_partner, qubit, fib_e in (
            (prev_partner, prev_epr, next_partner, prev_qubit, prev_fib_entry),
            (next_partner, next_epr, prev_partner, next_qubit, next_fib_entry),
        ):
            if new_epr is not None:
                partner.get_app(ProactiveForwarder).remote_swapped_eprs[new_epr.name] = new_epr

            su_msg: SwapUpdateMsg = {
                "cmd": "SWAP_UPDATE",
                "path_id": fib_entry["path_id"],
                "swapping_node": self.own.name,
                "partner": new_partner.name,
                "epr": old_epr.name,
                "new_epr": None if new_epr is None else new_epr.name,
            }
            self.send_msg(dest=partner, msg=su_msg, route=fib_e["path_vector"])

        # Release old qubits.
        for i, qubit in enumerate((prev_qubit, next_qubit)):
            qubit.fsm.to_release()
            simulator.add_event(QubitReleasedEvent(self.own, qubit, t=(simulator.tc + i * 1e-6), by=self))

    def handle_swap_update(self, msg: SwapUpdateMsg, fib_entry: FIBEntry):
        """Processes an SWAP_UPDATE signaling message from a neighboring node, updating local
        qubit state, releasing decohered pairs, or forwarding the update along the path.
        Handles both sequential and parallel swap scenarios, updates quantum memory with
        new EPRs when valid, and updates the qubit state.

        Parameters
        ----------
            msg: The SWAP_UPDATE message.
            fib_entry: FIB entry associated with path_id in the message.

        """
        if self.own.timing_mode == TimingModeEnum.SYNC and self.sync_current_phase != SignalTypeEnum.INTERNAL:
            log.debug(f"{self.own}: INT phase is over -> stop swaps")
            return

        _, sender_rank = find_index_and_swapping_rank(fib_entry, msg["swapping_node"])
        _, own_rank = find_index_and_swapping_rank(fib_entry, self.own.name)
        if own_rank < sender_rank:
            log.debug(f"### {self.own}: VERIFY -> rcvd SU from higher-rank node")
            return

        new_epr_name = msg["new_epr"]
        new_epr = None if new_epr_name is None else self.remote_swapped_eprs.pop(new_epr_name)

        epr_name = msg["epr"]
        qubit_pair = self.memory.get(key=epr_name)
        if qubit_pair is not None:
            qubit, _ = qubit_pair
            self.parallel_swappings.pop(epr_name, None)
            self._su_sequential(msg, fib_entry, qubit, new_epr, maybe_purif=(own_rank > sender_rank))
        elif own_rank == sender_rank and epr_name in self.parallel_swappings:
            self._su_parallel(msg, fib_entry, own_rank, new_epr)
        else:
            log.debug(f"### {self.own}: EPR {epr_name} decohered during SU transmissions")

    CLASSIC_SIGNALING_HANDLERS["SWAP_UPDATE"] = handle_swap_update

    def _su_sequential(
        self,
        msg: SwapUpdateMsg,
        fib_entry: FIBEntry,
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
            self.memory.read(address=qubit.addr)  # destructive reading
            qubit.fsm.to_release()
            # Inform LinkLayer that the memory qubit has been released.
            simulator.add_event(QubitReleasedEvent(self.own, qubit, t=simulator.tc, by=self))
            return

        # Update old EPR with new EPR (fidelity and partner).
        updated = self.memory.update(old_qm=msg["epr"], new_qm=new_epr)
        if not updated:
            raise Exception(f"{self.own}: EPR update failed | old={msg['epr']} , new={new_epr}")

        if maybe_purif:
            # If own rank is higher than sender rank but lower than new partner rank,
            # it is our turn to purify the qubit and progress toward swapping.
            qubit.purif_rounds = 0
            qubit.fsm.to_purif()
            partner = self.own.network.get_node(msg["partner"])
            self.qubit_is_purif(qubit, fib_entry, partner)

    def _su_parallel(self, msg: SwapUpdateMsg, fib_entry: FIBEntry, own_rank: int, new_epr: WernerStateEntanglement | None):
        """
        Process SWAP_UPDATE message during parallel swapping.
        """
        simulator = self.simulator
        (shared_epr, other_epr, my_new_epr) = self.parallel_swappings.pop(msg["epr"])
        _ = shared_epr

        # safety in statistical mux to avoid conflictual swappings on different paths
        if my_new_epr.tmp_path_ids is not None and msg["path_id"] not in my_new_epr.tmp_path_ids:
            if self.statistical_mux:
                log.debug(f"{self.own}: Conflictual parallel swapping in statistical mux -> silently ignore")
                return
            else:
                raise Exception(f"{self.own}: Unexpected conflictual parallel swapping")

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
            self.send_msg(dest=destination, msg=su_msg, route=fib_entry["path_vector"])
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
        if merged_epr is not None and new_epr.tmp_path_ids is not None:
            if not self.statistical_mux:  # dynamic EPR affectation
                assert new_epr.tmp_path_ids == other_epr.tmp_path_ids
                merged_epr.tmp_path_ids = list(new_epr.tmp_path_ids)
            else:  # statistical mux
                assert other_epr.tmp_path_ids is not None
                path_ids = select_common_element(new_epr.tmp_path_ids, other_epr.tmp_path_ids)
                if not path_ids:
                    raise Exception(f"Cannot select path ID from {new_epr.tmp_path_ids} and {other_epr.tmp_path_ids}")
                merged_epr.tmp_path_ids = path_ids

        # check EPR for non-isolated paths
        # if merged_epr is not None:
        #     endpoints = {merged_epr.src.name, merged_epr.dst.name}
        #     if not endpoints.issubset(set(fib_entry["path_vector"])):
        #         if self.isolate_paths:  # isolated-paths
        #             raise Exception("Unexpected conflictual parallel swapping")
        #         else:
        #             log.debug(
        #                 f"{self.own}: Ignored conflictual parallel swapping in non-isolated paths. "
        #                 "Caused by two nodes swapping a the same EPR with two different paths."
        #             )
        #             return

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
        self.send_msg(dest=destination, msg=su_msg, route=fib_entry["path_vector"])

        # Update records to support potential parallel swapping with "partner".
        _, p_rank = find_index_and_swapping_rank(fib_entry, partner.name)
        if own_rank == p_rank and merged_epr is not None:
            self.parallel_swappings[new_epr.name] = (new_epr, other_epr, merged_epr)

    def consume_and_release(self, qubit: MemoryQubit):
        """
        Consume an entangled qubit.
        """
        simulator = self.simulator

        _, qm = self.memory.read(address=qubit.addr, must=True)
        assert isinstance(qm, WernerStateEntanglement)
        assert qm.src is not None
        assert qm.dst is not None
        qubit.fsm.to_release()
        log.debug(f"{self.own}: consume EPR: {qm.name} -> {qm.src.name}-{qm.dst.name} | F={qm.fidelity}")

        self.cnt.n_consumed += 1
        self.cnt.consumed_sum_fidelity += qm.fidelity
        simulator.add_event(QubitReleasedEvent(self.own, qubit, t=simulator.tc, by=self))

    def send_msg(self, dest: Node, msg: Any, route: list[str]):
        own_idx = route.index(self.own.name)
        dest_idx = route.index(dest.name)

        nh = route[own_idx + 1] if dest_idx > own_idx else route[own_idx - 1]
        next_hop = self.own.network.get_node(nh)

        log.debug(f"{self.own.name}: send msg to {dest.name} via {next_hop.name} | msg: {msg}")

        cchannel = self.own.get_cchannel(next_hop)
        classic_packet = ClassicPacket(msg=msg, src=self.own, dest=dest)
        cchannel.send(classic_packet, next_hop=next_hop)

    def handle_sync_signal(self, signal_type: SignalTypeEnum):
        """Processes timing signals for SYNC mode. When receiving an INTERNAL phase start signal, all
        previously queued QubitEntangledEvent instances are processed. Updates the current
        synchronization phase to match the received signal type.

        Parameters
        ----------
            signal_type (SignalTypeEnum): The received synchronization signal.

        """
        if signal_type == SignalTypeEnum.EXTERNAL:
            self.remote_swapped_eprs.clear()
        elif signal_type == SignalTypeEnum.INTERNAL:
            # internal phase -> time to handle all entangled qubits
            log.debug(f"{self.own}: there are {len(self.waiting_qubits)} etg qubits to process")
            for event in self.waiting_qubits:
                self.qubit_is_entangled(event)
            self.waiting_qubits = []


def _can_enter_purif(own_name: str, partner_name: str) -> bool:
    """
    Evaluate if a qubit is eligible for purification, in statistical_mux only with limited support.

    - Any entangled qubit at intermediate node is always eligible.
    - Entangled qubit at end-node is eligible only if entangled with another end-node.
    """
    return (
        (own_name.startswith("R"))
        or (own_name.startswith("S") and partner_name.startswith("D"))
        or (own_name.startswith("D") and partner_name.startswith("S"))
    )


def select_common_element(list1: list[int], list2: list[int]) -> list[int] | None:
    set1 = set(list1)
    set2 = set(list2)

    # Case 1: one list has only one element
    if len(list1) == 1:
        return [list1[0]] if list1[0] in set2 else None
    if len(list2) == 1:
        return [list2[0]] if list2[0] in set1 else None

    # Case 2: both have more than one element
    common = list(set1 & set2)
    return common


def random_path_selector(fibs: list[FIBEntry]) -> int:
    return random.choice(fibs)["path_id"]
