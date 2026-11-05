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

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import Any, Final, TypedDict, Unpack, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.memory import QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.ll.counter import LinkLayerCounters
from mqns.network.ll.entangle import LLEntangleProc, LLGroup, LLNotifyEvent, ReserveMsg
from mqns.network.ll.entangle_2nd import LLEntangleProc2nd, LLGroup2nd, LLNotify2ndEvent
from mqns.network.ll.entangle_pri import LLEntangleProcPri, LLGroupPri, LLNotifyPriEvent
from mqns.network.ll.event import PathActivateEvent, PathDeactivateEvent, QubitEntangledEvent, QubitReleasedEvent
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.simulator import event_handler
from mqns.utils import rng, unwrap_cast


class LinkLayerInitKwargs(TypedDict, total=False):
    frequency: float
    """Entanglement source frequency, defaults to ``1_000_000``."""
    tau_0: float
    """Local operation delay in seconds for emitting and absorbing photon, defaults to ``0.0``."""
    cnt_history: int
    """
    How many recent entanglement to keep in per-channel history, default is ``0``.
    A positive value enables online collection of channel entanglement rates.
    """


class LinkLayer(ClassicCommandDispatcherMixin, Application[QNode]):
    """
    Network function for creating elementary entanglements over qchannels.
    It equips a QNode and is activated from the forwarding function (e.g., ProactiveForwarder).
    """

    entangle_by_ch: Final[MutableMapping[QuantumChannel, LLEntangleProc]]
    entangle_by_peer: Final[MutableMapping[str, LLEntangleProc]]
    entangle_procs: Final[Iterable[LLEntangleProc[LLGroup[Any]]]]
    group_by_mq: Final[MutableMapping[int, LLGroup]]

    def __init__(self, **kwargs: Unpack[LinkLayerInitKwargs]):
        super().__init__()
        self.reset_time = 1 / kwargs.get("frequency", 1e6)
        self.tau_0 = kwargs.get("tau_0", 0.0)

        self.entangle_by_ch = {}
        self.entangle_by_peer = {}
        self.entangle_procs = self.entangle_by_peer.values()
        self.group_by_mq = {}

        self._cnt_deleted_channels = LinkLayerCounters()
        self._cnt_history = kwargs.get("cnt_history", 0)

    @override
    def install(self, node) -> None:
        self._application_install(node, QNode)
        self.memory = self.node.memory
        """Quantum memory of the node."""

    @property
    def cnt(self) -> LinkLayerCounters:
        """Retrieve total counters."""
        r = LinkLayerCounters()
        r += self._cnt_deleted_channels
        for proc in self.entangle_procs:
            if type(proc) is LLEntangleProcPri:
                r += proc.cnt
        return r

    def cnt_channel(self, ch: QuantumChannel) -> LinkLayerCounters | None:
        """
        Retrieve per-channel counters.

        Args:
            ch: A quantum channel.

        Returns:
            Counters of an active channel where this node is primary.
            None if the channel is inactive or this node is secondary.

        Note:
            If a channel is recently deactivated but not yet deleted, its counters would still be returned.
            If the same channel is reactivated, the counters would continue accumulating without reset.
        """
        proc = self.entangle_by_ch.get(ch)
        if type(proc) is LLEntangleProcPri:
            return proc.cnt
        return None

    @sync_phase_handler(TimingPhase.EXTERNAL, True)
    def sync_external_enter(self) -> None:
        for proc in self.entangle_procs:
            for grp in proc.groups_active.values():
                grp.sync_external_enter()

    @sync_phase_handler(TimingPhase.EXTERNAL, False)
    def sync_external_exit(self) -> None:
        for proc in self.entangle_procs:
            for grp in proc.groups_active.values():
                grp.sync_external_exit()

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self) -> None:
        # Clear existing memory qubits.
        self.memory.clear()

    @staticmethod
    def _gid_from_path_id(path_id: int | None) -> int:
        if path_id is None:
            return -9
        return path_id

    @event_handler
    def path_activate(self, event: PathActivateEvent) -> None:
        """
        Handle path activation command.
        """
        event.cancel()
        ch = event.qchannel
        path_id = event.path_id

        # Find or create the channel entangle procedure.
        LLEntangleProcTyp = LLEntangleProcPri if event.is_primary else LLEntangleProc2nd
        proc = self.entangle_by_ch.get(ch)
        if not proc:
            proc = LLEntangleProcTyp(self, ch)
            self.entangle_by_ch[ch] = self.entangle_by_peer[ch.find_peer(self.node).name] = proc
        elif type(proc) is not LLEntangleProcTyp:
            raise ValueError(f"{event}: channel already activated with opposite role in {proc}")

        # Activate the group within the channel.
        gid = self._gid_from_path_id(path_id)
        proc.group_activate(gid, [q for q, _ in self.memory.find(lambda q, _: q.path_id == path_id, qchannel=ch)])

    @event_handler
    def path_deactivate(self, event: PathDeactivateEvent) -> None:
        """
        Handle path deactivation command.
        """
        event.cancel()

        # Find the channel entangle procedure.
        try:
            proc = self.entangle_by_ch[event.qchannel]
        except LookupError:
            raise ValueError(f"{event}: channel not activated") from None

        # Deactivate the group within the channel.
        gid = self._gid_from_path_id(event.path_id)
        proc.group_deactivate(gid)

    def channel_deactivate(self, proc: LLEntangleProc) -> None:
        """
        Delete a channel, invoked by ``LLEntangleProc._group_delete()``.
        """
        del self.entangle_by_ch[proc.ch]
        del self.entangle_by_peer[proc.peer.name]
        if type(proc) is LLEntangleProcPri:
            proc.log_debug("CHANNEL_DEACTIVATE | %s", proc.cnt)
            self._cnt_deleted_channels += proc.cnt
        else:
            proc.log_debug("CHANNEL_DEACTIVATE")

    def _dispatch_reserve_msg[P: LLEntangleProc, G: LLGroup](
        self,
        pkt: ClassicPacket,
        msg: Mapping,
        typ: type[P],
        meth: Callable[[G, ReserveMsg], None],
        allow_draining=False,
    ) -> None:
        req = ReserveMsg(**msg)
        pkt_from = pkt.src.name

        if not self.node.timing.is_external():
            self.log_debug("%s from=%s ignored reason=not-external-phase", req, pkt_from)
            return

        proc = self.entangle_by_peer.get(pkt_from)
        if type(proc) is not typ:
            self.log_debug("%s from=%s ignored reason=no-channel", req, pkt_from)
            return
        proc = unwrap_cast(proc)

        grp = proc.groups_active.get(req.gid)
        if grp is None and allow_draining:
            grp = proc.groups_draining.get(req.gid)
        if grp is None:
            self.log_debug("%s from=%s ignored reason=no-group", req, pkt_from)
            return

        meth(grp, req)

    @classic_cmd_handler("RESERVE_REQ")
    def handle_reserve_req(self, pkt: ClassicPacket, msg: Mapping) -> None:
        """
        Secondary node handles ``RESERVE_REQ`` message from the primary node.
        """
        self._dispatch_reserve_msg(pkt, msg, LLEntangleProc2nd, LLGroup2nd.handle_reserve_req)

    @classic_cmd_handler("RESERVE_ABORT")
    def handle_reserve_abort(self, pkt: ClassicPacket, msg: Mapping) -> None:
        """
        Secondary node handles ``RESERVE_ABORT`` message from the primary node.
        """
        self._dispatch_reserve_msg(pkt, msg, LLEntangleProc2nd, LLGroup2nd.handle_reserve_abort, allow_draining=True)

    @classic_cmd_handler("RESERVE_RES")
    def handle_reserve_res(self, pkt: ClassicPacket, msg: Mapping) -> None:
        """
        Primary node handles ``RESERVE_RES`` message from the secondary node.
        """
        self._dispatch_reserve_msg(pkt, msg, LLEntangleProcPri, LLGroupPri.handle_reserve_res)

    def _calc_attempt(self, qchannel: QuantumChannel) -> int:
        return rng.geometric(qchannel.link_arch.success_prob)

    @event_handler
    def _notify_pri(self, event: LLNotifyPriEvent) -> None:
        self._notify_entangle(event)

    @event_handler
    def _notify_2nd(self, event: LLNotify2ndEvent) -> None:
        self._notify_entangle(event)

    def _notify_entangle(self, event: LLNotifyEvent) -> None:
        event.cancel()
        assert self.node.timing.is_external()
        mq = self.memory.write(event.key, event.epr)

        self.log_debug("%s EPR_DELIVER mq-addr=%s path=%s", event, mq.addr, mq.path_id)

        mq.state = QubitState.ENTANGLED0
        self.simulator.sched(QubitEntangledEvent(self.node, event.peer, mq, t=event.t))

    @event_handler
    def handle_release(self, event: QubitReleasedEvent) -> None:
        mq = event.qubit
        mq.state = QubitState.RAW

        grp = self.group_by_mq.get(mq.addr)
        if not grp:
            self.log_debug("%s ignored reason=no-group", event)
            return

        grp.handle_release(event)
