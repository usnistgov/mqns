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

from collections.abc import Iterable, MutableMapping
from typing import Final, TypedDict, Unpack, cast, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.memory import QuantumMemory, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.ll.counter import LinkLayerCounters
from mqns.network.ll.entangle import LinkLayerEntangle, LinkLayerEntangle2nd, LinkLayerEntanglePri
from mqns.network.ll.event import PathActivateEvent, PathDeactivateEvent, QubitEntangledEvent, QubitReleasedEvent
from mqns.network.ll.message import ReserveMsg, ReserveParsedMsg
from mqns.network.ll.task import ActivePath, Entangle2ndEvent, EntanglePriEvent
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.simulator import event_handler, func_to_event
from mqns.utils import rng, unwrap, unwrap_cast


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

    entangle_by_ch: Final[MutableMapping[QuantumChannel, LinkLayerEntangle]]
    entangle_by_peer: Final[MutableMapping[str, LinkLayerEntangle]]
    entangle_procs: Final[Iterable[LinkLayerEntangle]]

    def __init__(self, **kwargs: Unpack[LinkLayerInitKwargs]):
        super().__init__()
        self.reset_time = 1 / kwargs.get("frequency", 1e6)
        self.tau_0 = kwargs.get("tau_0", 0.0)

        self.entangle_by_ch = {}
        self.entangle_by_peer = {}
        self.entangle_procs = self.entangle_by_peer.values()

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
            if type(proc) is LinkLayerEntanglePri:
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
        if type(proc) is LinkLayerEntanglePri:
            return proc.cnt
        return None

    @sync_phase_handler(TimingPhase.EXTERNAL, True)
    def sync_external_enter(self) -> None:
        for proc in self.entangle_procs:
            proc.sync_external_enter()

    @sync_phase_handler(TimingPhase.EXTERNAL, False)
    def sync_external_exit(self) -> None:
        for proc in self.entangle_procs:
            proc.sync_external_exit()

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self) -> None:
        # Clear existing memory qubits.
        self.memory.clear()

    @event_handler
    def activate(self, event: PathActivateEvent) -> None:
        """
        Handle path activation command.
        """
        event.cancel()
        ch = event.qchannel
        is_primary = event.is_primary
        path_id = event.path_id

        # Find or create the channel entangle procedure.
        LinkLayerEntangleTyp = LinkLayerEntanglePri if is_primary else LinkLayerEntangle2nd
        proc = self.entangle_by_ch.get(ch)
        if not proc:
            proc = LinkLayerEntangleTyp(self, ch)
            self.entangle_by_ch[ch] = self.entangle_by_peer[ch.find_peer(self.node).name] = proc
        elif type(proc) is not LinkLayerEntangleTyp:
            raise RuntimeError(f"{event}: channel already activated with opposite role in {proc}")

        # Find or insert the ActivePath record.
        ap = proc.all_paths.get(path_id)
        if not ap:
            ap = ActivePath(path_id, is_primary)
            proc.all_paths[path_id] = ap

        # Mark the path as live.
        ap.deleting = False
        ap.delete_event.cancel()
        proc.live_paths[path_id] = ap
        addrs = [q.addr for q, _ in self.memory.find(QuantumMemory.predicate_all, qchannel=ch)]
        proc.log_debug("PATH_ACTIVATE %s has-addrs=%s", ap, addrs)

        # Start the reservations if the network is in ASYNC timing mode.
        if type(proc) is LinkLayerEntanglePri and self.node.timing.is_async():
            proc.run_channel(ap)

    @event_handler
    def deactivate(self, event: PathDeactivateEvent) -> None:
        """
        Handle path deactivation command.
        """
        now = event.t
        event.cancel()
        path_id = event.path_id

        # Find the channel entangle procedure.
        proc = self.entangle_by_ch[event.qchannel]

        # Find the ActivePath record and mark as pending deletion.
        ap = proc.all_paths[path_id]
        ap.deleting = True
        del proc.live_paths[path_id]
        t_delete = now + self.memory.t_cohere
        self.simulator.sched(delete_event := func_to_event(t_delete, self._deactivate_delete, proc, ap))
        ap.delete_event.set(delete_event)

        if type(proc) is LinkLayerEntangle2nd:
            # On the secondary node, no more reservations would be accepted.
            ap.ireq_queue.clear()
            self.log_debug("PATH_DEACTIVATE_2nd %s.%s t_delete=%s", proc, ap, t_delete)
            return

        # On the primary node, no more attempts may start.
        # Since the skip-ahead sampling method only cares about the successful attempt,
        # this logic allows the entanglement to continue if the successful attempt has started
        # or is starting in the current time slot, otherwise it is aborted.
        tasks_aborted: list[str] = []
        tasks_uncancelable: list[str] = []
        for key, task in ap.oreq_table.items():
            if task.k == -1:  # reservation has not completed
                tasks_aborted.append(key)
                continue
            if task.t_success <= now:  # successful attempt has already started
                tasks_uncancelable.append(key)
                continue
            tasks_aborted.append(key)
            unwrap_cast(task.notify_pri_event).cancel()
            unwrap_cast(task.notify_2nd_event).cancel()

        for key in tasks_aborted:
            task = ap.oreq_table.pop(key)
            task.mq.state = QubitState.RAW
            msg = ReserveMsg(cmd="RESERVE_ABORT", key=key, path_id=path_id)
            self.node.send_cpacket(proc.peer, ClassicPacket(msg, src=self.node, dest=proc.peer))

        self.log_debug(
            "PATH_DEACTIVATE_pri %s.%s t_delete=%s abort-tasks=%s uncancelable-tasks=%s",
            proc,
            ap,
            t_delete,
            tasks_aborted,
            tasks_uncancelable,
        )

    def _deactivate_delete(self, proc: LinkLayerEntangle, ap: ActivePath) -> None:
        """
        Part of ``PathDeactivateEvent`` handler: delayed deletion of ActivePath record.
        """
        path_id = ap.path_id

        # If the path has been re-inserted, the delete_event should have been canceled.
        assert ap.deleting, f"{self}: {proc}.{ap}.deleting is False"

        # There should be nothing pending on the ActivePath record.
        if is_primary := type(proc) is LinkLayerEntanglePri:
            assert len(ap.oreq_table) == 0, f"{self}: {proc}.{ap}.oreq_table = {ap.oreq_table}"
        else:
            assert len(ap.ireq_queue) == 0, f"{self}: {proc}.{ap}.ireq_queue = {ap.ireq_queue}"

        # There should be no qubits owned by LinkLayer.
        locked_qubits = list(
            self.memory.find(
                lambda q, _: q.state in (QubitState.ACTIVE, QubitState.RESERVED) and q.path_id == path_id, qchannel=proc.ch
            )
        )
        assert not locked_qubits, f"{self}: {proc}.{ap} has locked qubits: {locked_qubits}"

        # Delete the path from the channel.
        del proc.all_paths[path_id]
        self.log_debug("PATH_DEACTIVATE %s.%s deleted", proc, ap)
        if len(proc.all_paths) > 0:
            return

        # If the channel has no more active paths, delete the channel.
        del self.entangle_by_ch[proc.ch]
        del self.entangle_by_peer[proc.peer.name]
        if is_primary:
            self._cnt_deleted_channels += proc.cnt
        self.log_debug("CHANNEL_DEACTIVATE %s", proc)

    @classic_cmd_handler("RESERVE_ABORT")
    def handle_reserve_abort(self, pkt: ClassicPacket, msg: ReserveMsg) -> None:
        """
        Handle ``RESERVE_ABORT`` message from the primary node.
        """
        req = ReserveParsedMsg.new(pkt, msg)
        if not self.node.timing.is_external():
            self.log_debug("%s ignored reason=not-external-phase", req)
            return None

        # Reset the memory, if present.
        if (mq_tuple := self.memory.read(req.key)) is None:
            self.log_debug("%s ignored reason=qubit-not-found", req)
            return
        mq, qm = mq_tuple

        self.log_debug("%s canceling addr=%s", req, mq.addr)
        assert qm is None
        assert mq.state in (QubitState.ACTIVE, QubitState.RESERVED)
        mq.state = QubitState.RAW

        # It's unnecessary to delete the key from ireq_queue here,
        # because the path deletion logic on the secondary node would clear ireq_queue.

    def _handle_reserve_msg[P: LinkLayerEntangle](self, req: ReserveParsedMsg, typ: type[P]) -> P | None:
        if not self.node.timing.is_external():
            self.log_debug("%s ignored reason=not-external-phase", req)
            return None

        proc = self.entangle_by_peer.get(req.src)
        if type(proc) is not typ:
            self.log_debug("%s ignored reason=no-active-channel", req)
            return None

        return proc

    @classic_cmd_handler("RESERVE_REQ")
    def handle_reserve_req(self, pkt: ClassicPacket, msg: ReserveMsg) -> None:
        """
        Handle ``RESERVE_REQ`` message from the primary node.
        """
        proc = self._handle_reserve_msg(req := ReserveParsedMsg.new(pkt, msg), LinkLayerEntangle2nd)
        if proc:
            proc.handle_reserve_req(req)

    @classic_cmd_handler("RESERVE_RES")
    def handle_reserve_res(self, pkt: ClassicPacket, msg: ReserveMsg) -> None:
        """
        Handle ``RESERVE_RES`` message from the secondary node.
        """
        proc = self._handle_reserve_msg(req := ReserveParsedMsg.new(pkt, msg), LinkLayerEntanglePri)
        if proc:
            proc.handle_reserve_res(req)

    def _calc_attempt(self, qchannel: QuantumChannel) -> int:
        return rng.geometric(qchannel.link_arch.success_prob)

    @event_handler
    def _notify_pri(self, event: EntanglePriEvent) -> None:
        del event.task.ap.oreq_table[event.key]
        cast(LinkLayerEntanglePri, event.task.ac).cnt.save_etg(event.task)
        self._notify_entangle("pri", event)

    @event_handler
    def _notify_2nd(self, event: Entangle2ndEvent) -> None:
        self._notify_entangle("2nd", event)

    def _notify_entangle(self, own_role: str, event: EntanglePriEvent | Entangle2ndEvent) -> None:
        event.cancel()
        assert self.node.timing.is_external()
        partner = event.partner
        key = event.key
        epr = event.epr
        mq = self.memory.write(key, epr)

        self.log_debug(
            "EPR_DELIVER_%s %s partner=%s key=%s addr=%s path=%s",
            own_role,
            epr.name,
            partner.name,
            key,
            mq.addr,
            mq.path_id,
        )
        assert epr.decohere_time > self.simulator.tc

        mq.state = QubitState.ENTANGLED0
        self.simulator.sched(QubitEntangledEvent(self.node, partner, mq, t=self.simulator.tc))

    @event_handler
    def handle_release(self, event: QubitReleasedEvent) -> None:
        mq = event.qubit
        mq.state = QubitState.RAW

        proc = self.entangle_by_ch.get(unwrap(mq.qchannel))
        if not proc:
            self.log_debug("%s ignored reason=no-active-channel", event)
            return

        proc.handle_release(event)
