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

from typing import TypedDict, Unpack, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.memory import MemoryQubit, QuantumMemory, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.delay import ConstantDelayModel
from mqns.network.ll.counter import LinkLayerCounters
from mqns.network.ll.event import PathActivateEvent, PathDeactivateEvent, QubitEntangledEvent, QubitReleasedEvent
from mqns.network.ll.message import ReserveMsg, ReserveParsedMsg
from mqns.network.ll.task import ActiveChannel, ActivePath, Entangle2ndEvent, EntanglePriEvent, EntangleTask
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.simulator import event_handler, func_to_event
from mqns.utils import AutoIncrementIdentifier, rng, unwrap, unwrap_cast

_AUTOID = AutoIncrementIdentifier("llk_")
"""
Automatically assigned ``RESERVE_REQ["key"]`` name.
"""


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

    def __init__(self, **kwargs: Unpack[LinkLayerInitKwargs]):
        super().__init__()
        self.reset_time = 1 / kwargs.get("frequency", 1e6)
        self.tau_0 = kwargs.get("tau_0", 0.0)

        self.channels: dict[str, ActiveChannel] = {}
        """
        Active channel table.
        Key is partner node name.
        """

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
        for ac in self.channels.values():
            if ac.is_primary:
                r += ac.cnt
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
        partner = ch.find_peer(self.node)
        ac = self.channels.get(partner.name)
        if ac and ac.is_primary:
            return ac.cnt
        return None

    @sync_phase_handler(TimingPhase.EXTERNAL, True)
    def sync_external_enter(self) -> None:
        # Start reservation for each active channel where this node is primary.
        for ac in self.channels.values():
            if ac.is_primary:
                self.run_channel(ac)

    @sync_phase_handler(TimingPhase.EXTERNAL, False)
    def sync_external_exit(self) -> None:
        for ac in self.channels.values():
            if ac.is_primary:
                # Clear incomplete reservations.
                for ap in ac.paths.values():
                    ap.oreq_table.clear()
            else:
                # Clear unsatisfied reservation requests.
                for ap in ac.paths.values():
                    ap.ireq_queue.clear()

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
        partner = ch.find_peer(self.node)
        path_id = event.path_id

        # Find or insert the ActiveChannel record.
        ac = self.channels.get(partner.name)
        if not ac:
            rtt = self.simulator.time(sec=ConstantDelayModel.extract(ch.delay) * 2)
            if event.is_primary:
                ac = ActiveChannel(ch, partner, LinkLayerCounters(history=self._cnt_history, rtt=rtt))
            else:
                ac = ActiveChannel(ch, partner, None)
            self.channels[partner.name] = ac

            self._activate_channel(ac)
        elif ac.is_primary is not event.is_primary:
            raise RuntimeError(f"{event}: channel already activated with role={PathActivateEvent.ROLE_STR[ac.is_primary]}")

        # Find or insert the ActivePath record.
        ap = ac.paths.get(path_id)
        if not ap:
            ap = ActivePath(path_id, ac.is_primary)
            ac.paths[path_id] = ap

        # Increment the insertion count.
        ap.insertion_count += 1
        if ap.insertion_count > 1:
            return

        # If the path is new, mark it as live.
        ac.live_paths.add(path_id)
        ap.delete_event.cancel()
        addrs = [q.addr for q, _ in self.memory.find(QuantumMemory.predicate_all, qchannel=ch)]
        self.log_debug("PATH_ACTIVATE_%s %s.%s has-addrs=%s", PathActivateEvent.ROLE_STR[ac.is_primary], ac, ap, addrs)

        # Start the reservations if the network is in ASYNC timing mode.
        if ac.is_primary and self.node.timing.is_async():
            self.run_channel(ac, ap)

    def _activate_channel(self, ac: ActiveChannel) -> None:
        """
        Part of ``PathActivateEvent`` handler: activate a previously idle channel.
        """
        ch = ac.qchannel
        if not ac.is_primary:
            self.log_debug("CHANNEL_ACTIVATE_2nd %s ch=%s", ac, ch.name)
            return

        la = ch.link_arch

        # link_arch.set() may be called multiple times in several situations:
        # - qchannel is activated, deactivated, and re-activated.
        # - qchannel is activated from both sides due to shared paths in opposite directions.
        # Nevertheless, we assume every LinkLayer instance has the same parameters,
        # so that the parameters saved into LinkArch are the same every time.
        la.set(
            time_accuracy=self.simulator.accuracy,
            ch=ch,
            reset_time=self.reset_time,
            tau_0=self.tau_0,
            epr_type=self.node.network.epr_type,
            store_decays=(self.memory.time_decay, ac.partner.memory.time_decay),
        )

        epr_tpl = la.make_epr(self.simulator.ts, self.node, ac.partner, key=None)

        self.log_debug(
            "CHANNEL_ACTIVATE_pri %s ch=%s link_arch=%s attempt_interval=%s t_notify=%s,%s | template %s",
            ac,
            ch.name,
            la.name,
            la.attempt_interval,
            la.d_notify_pri,
            la.d_notify_2nd,
            epr_tpl,
        )

    @event_handler
    def deactivate(self, event: PathDeactivateEvent) -> None:
        """
        Handle path deactivation command.
        """
        now = event.t
        event.cancel()
        ch = event.qchannel
        partner = ch.find_peer(self.node)
        path_id = event.path_id

        # Find the ActiveChannel record.
        ac = self.channels[partner.name]
        assert ac.qchannel is ch

        # Find the ActivePath record and decrement its insertion count.
        ap = ac.paths[path_id]
        ap.insertion_count -= 1
        if ap.insertion_count > 0:
            return

        # When a path reaches insertion_count==0, it is marked as pending deletion.
        ac.live_paths.remove(path_id)
        t_delete = now + self.memory.t_cohere
        self.simulator.sched(delete_event := func_to_event(t_delete, self._deactivate_delete, ac, ap))
        ap.delete_event.set(delete_event)

        if not ac.is_primary:
            # On the secondary node, no more reservations would be accepted.
            ap.ireq_queue.clear()
            self.log_debug("PATH_DEACTIVATE_2nd %s.%s t_delete=%s", ac, ap, t_delete)
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
            self.node.send_cpacket(ac.partner, ClassicPacket(msg, src=self.node, dest=ac.partner))

        self.log_debug(
            "PATH_DEACTIVATE_pri %s.%s t_delete=%s abort-tasks=%s uncancelable-tasks=%s",
            ac,
            ap,
            t_delete,
            tasks_aborted,
            tasks_uncancelable,
        )

    def _deactivate_delete(self, ac: ActiveChannel, ap: ActivePath) -> None:
        """
        Part of ``PathDeactivateEvent`` handler: delayed deletion of ActivePath record.
        """
        # If the path has been re-inserted, the delete_event should have been canceled.
        assert ap.insertion_count == 0, f"{self}: {ac}.{ap}.insertion_count={ap.insertion_count}"

        ch = ac.qchannel
        partner = ac.partner
        path_id = ap.path_id

        # There should be nothing pending on the ActivePath record.
        if ac.is_primary:
            assert len(ap.oreq_table) == 0, f"{self}: {ac}.{ap}.oreq_table = {ap.oreq_table}"
        else:
            assert len(ap.ireq_queue) == 0, f"{self}: {ac}.{ap}.ireq_queue = {ap.ireq_queue}"

        # There should be no qubits owned by LinkLayer.
        locked_qubits = list(
            self.memory.find(
                lambda q, _: q.state in (QubitState.ACTIVE, QubitState.RESERVED) and q.path_id == path_id, qchannel=ch
            )
        )
        assert not locked_qubits, f"{self}: {ac}.{ap} has locked qubits: {locked_qubits}"

        # Delete the path from the channel.
        del ac.paths[path_id]
        self.log_debug("PATH_DEACTIVATE_%s %s.%s deleted", PathActivateEvent.ROLE_STR[ac.is_primary], ac, ap)
        if len(ac.paths) > 0:
            return

        # If the channel has no more active paths, delete the channel.
        del self.channels[partner.name]
        if ac.is_primary:
            self._cnt_deleted_channels += ac.cnt
        self.log_debug("CHANNEL_DEACTIVATE_%s %s partner=%s", PathActivateEvent.ROLE_STR[ac.is_primary], ch.name, partner.name)

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

    def run_channel(self, ac: ActiveChannel, ap: ActivePath | None = None) -> None:
        """
        Start entanglement generation on qubits assigned to a channel.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            ap: If set, only consider qubits allocated to a specified path.
        """
        qubits = self.memory.find(
            (lambda q, _: q.state is QubitState.RAW and q.path_id in ac.live_paths)
            if ap is None
            else (lambda q, _: q.state is QubitState.RAW and q.path_id == ap.path_id),
            qchannel=ac.qchannel,
        )
        for q, v in qubits:
            assert q.key is None
            assert v is None
            self.start_reservation(ac, None, q)

    def start_reservation(self, ac: ActiveChannel, ap: ActivePath | None, mq: MemoryQubit) -> None:
        """
        Start entanglement generation on a qubit assigned to a channel.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            ap: ActivePath record, if known.
            mq: The qubit, which must be in RAW state.
        """
        assert ac.is_primary
        ap = ap or ac.paths[mq.path_id]
        assert ap.insertion_count > 0

        # Generate a unique reservation key to represent this entanglement generation protocol execution
        # and the entanglement, which is valid until the entanglement is released.
        mq.key = _AUTOID()
        mq.state = QubitState.ACTIVE

        # Remember the pending reservation in ActivePath record.
        ap.oreq_table[mq.key] = EntangleTask(ac, ap, mq)
        self.log_debug(
            "RESERVE_REQ(%s:%s, %s) sent addr=%s partner=%s", self.node.name, mq.path_id, mq.key, mq.addr, ac.partner.name
        )

        # Transmit RESERVE_REQ message to the secondary node.
        msg = ReserveMsg(cmd="RESERVE_REQ", path_id=mq.path_id, key=mq.key)
        self.node.send_cpacket(ac.partner, ClassicPacket(msg, src=self.node, dest=ac.partner))

    def _find_active_channel_path(self, req: ReserveParsedMsg) -> tuple[ActiveChannel, ActivePath] | None:
        if not self.node.timing.is_external():
            self.log_debug("%s ignored reason=not-external-phase", req)
            return None

        ac = self.channels.get(req.src)
        if ac is None:
            self.log_debug("%s ignored reason=no-active-channel", req)
            return None

        ap = ac.paths.get(req.path_id)
        if ap is None or ap.insertion_count == 0:
            self.log_debug("%s ignored reason=no-active-path", req)
            return None

        return ac, ap

    @classic_cmd_handler("RESERVE_REQ")
    def handle_reserve_req(self, pkt: ClassicPacket, msg: ReserveMsg) -> None:
        """
        Handle ``RESERVE_REQ`` message from the primary node.
        """
        req = ReserveParsedMsg.new(pkt, msg)
        if (ac_ap := self._find_active_channel_path(req)) is None:
            return
        ac, ap = ac_ap
        assert not ac.is_primary

        if len(ap.ireq_queue) > 0 or not self.try_accept_reservation(ac, ap, req.key):
            # If the queue is non-empty, try_accept_reservation() would not have succeeded.
            # If try_accept_reservation() was unsuccessful, enqueue the request.
            ap.ireq_queue.append(req.key)

    def try_accept_reservation(self, ac: ActiveChannel, ap: ActivePath, key: str, *, hint: MemoryQubit | None = None) -> bool:
        """
        Accept a reservation if a qubit is available.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is secondary.
            ap: ActivePath record of the path to which the qubit is allocated.
            key: Reservation key from a received RESERVE_RES message.
            hint: If provided, it will be used, bypassing the search; it must be in RAW state.

        Returns:

            * True if the request is accepted and ``RESERVE_RES`` is sent.
              Caller is responsible for dequeuing the request if it was in a queue.
            * False if the request is not accepted.
              Caller is responsible for enqueuing the request.
        """
        assert not ac.is_primary
        assert ap.insertion_count > 0

        if hint:
            mq = hint
            assert mq.state is QubitState.RAW
            assert mq.path_id == ap.path_id
        else:
            mq, _ = next(
                self.memory.find(
                    lambda q, _: (
                        q.state is QubitState.RAW  # currently unoccupied and not part of an active reservation
                        and q.path_id == ap.path_id  # allocated to the path_id, if MuxScheme uses path_id
                    ),
                    qchannel=ac.qchannel,  # assigned to the quantum channel
                ),
                (None, None),
            )

        if mq is None:
            return False

        self.log_debug("RESERVE_REQ(%s:%s, %s) accepted addr=%s", ac.partner.name, ap.path_id, key, mq.addr)
        mq.state = QubitState.RESERVED
        mq.key = key
        msg = ReserveMsg(cmd="RESERVE_RES", path_id=ap.path_id, key=key)
        self.node.send_cpacket(ac.partner, ClassicPacket(msg, src=self.node, dest=ac.partner))
        return True

    @classic_cmd_handler("RESERVE_RES")
    def handle_reserve_res(self, pkt: ClassicPacket, msg: ReserveMsg) -> None:
        """
        Handle ``RESERVE_RES`` message from the secondary node.
        """
        req = ReserveParsedMsg.new(pkt, msg)
        if (ac_ap := self._find_active_channel_path(req)) is None:
            return
        ac, ap = ac_ap
        assert ac.is_primary

        task = ap.oreq_table.get(req.key, None)
        if task is None:
            self.log_debug("%s ignored reason=no-active-task", req)
            return

        mq = task.mq
        assert mq.key == req.key
        mq.state = QubitState.RESERVED
        self.log_debug("%s processed addr=%s", req, mq.addr)
        self.generate_entanglement(ac, ap, task)

    def generate_entanglement(self, ac: ActiveChannel, ap: ActivePath, task: EntangleTask) -> None:
        """
        Schedule a successful entanglement attempt using skip-ahead sampling.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            ap: ActivePath record of the path to which the qubit is allocated.
            mq: The memory qubit used for this attempt.
        """
        assert ac.is_primary
        assert ap.insertion_count > 0
        assert task.k == -1
        ch = ac.qchannel
        la = ch.link_arch
        key = task.key

        # Calculate which attempt would succeed (1-based).
        task.k = self._calc_attempt(ch)

        # Calculate when would the k-th attempt (1-based) succeed.
        # TODO space out EPRs on a qchannel by attempt_interval or qchannel.bandwidth
        task.t_reserve = self.simulator.tc
        task.t_success = task.t_reserve + la.attempt_interval * (task.k - 1)
        t_notify_pri = task.t_success + la.d_notify_pri
        t_notify_2nd = task.t_success + la.d_notify_2nd
        task.t_finish = max(t_notify_pri, t_notify_2nd)

        # If the network uses SYNC timing mode but the successful attempt would exceed the current EXTERNAL phase,
        # the EPR would not arrive in time, and therefore is not scheduled.
        if not self.node.timing.is_external(task.t_finish):
            self.log_debug(
                "EPR_SKIP partner=%s key=%s attempts=%s t_success=%s t_notify=%s,%s reason=beyond-external-phase",
                ac.partner.name,
                key,
                task.k,
                task.t_success,
                t_notify_pri,
                t_notify_2nd,
            )
            return

        epr = la.make_epr(task.t_success, self.node, ac.partner, key=key)

        # If the network uses ASYNC timing mode or the successful attempt can complete within the current EXTERNAL phase,
        # schedule the EPR arrival events on both nodes.
        self.log_debug(
            "EPR_PREPARE partner=%s key=%s attempts=%s epr=%s t_success=%s t_notify=%s,%s",
            ac.partner.name,
            key,
            task.k,
            epr.name,
            task.t_success,
            t_notify_pri,
            t_notify_2nd,
        )

        self.simulator.sched(notify_pri := EntanglePriEvent(key, epr, task, t=t_notify_pri))
        self.simulator.sched(notify_2nd := Entangle2ndEvent(key, epr, t=t_notify_2nd))
        task.notify_pri_event = notify_pri
        task.notify_2nd_event = notify_2nd

    def _calc_attempt(self, qchannel: QuantumChannel) -> int:
        return rng.geometric(qchannel.link_arch.success_prob)

    @event_handler
    def _notify_pri(self, event: EntanglePriEvent) -> None:
        del event.task.ap.oreq_table[event.key]
        event.task.ac.cnt.save_etg(event.task)
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

        partner = unwrap(mq.qchannel).find_peer(self.node)

        ac = self.channels.get(partner.name)
        if ac is None:
            self.log_debug("%s ignored reason=no-active-channel", event)
            return

        ap = ac.paths.get(mq.path_id)
        if ap is None or ap.insertion_count == 0:
            self.log_debug("%s ignored reason=no-active-path", event)
            return

        if ac.is_primary:
            self.log_debug("%s processed role=primary", event)
            if event.is_decoh:
                ac.cnt.n_decoh += 1
            if self.node.timing.is_async():
                self.start_reservation(ac, ap, mq)
        else:
            self.log_debug("%s processed role=secondary", event)
            if ap.ireq_queue and self.try_accept_reservation(ac, ap, ap.ireq_queue[0], hint=mq):
                ap.ireq_queue.popleft()
