from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, final, override

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.memory import MemoryQubit, QuantumMemory, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.delay import ConstantDelayModel
from mqns.network.ll.counter import LinkLayerCounters
from mqns.network.ll.event import QubitReleasedEvent
from mqns.network.ll.message import ReserveMsg, ReserveParsedMsg
from mqns.network.ll.task import ActiveChannel, ActivePath, Entangle2ndEvent, EntanglePriEvent, EntangleTask
from mqns.simulator import Simulator
from mqns.utils import AutoIncrementIdentifier, LogSelfMixin, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer

_AUTOID = AutoIncrementIdentifier("llk_")
"""
Automatically assigned ``RESERVE_REQ["key"]`` name.
"""


class LinkLayerEntangle(LogSelfMixin):
    """
    Entanglement generation procedure of one quantum channel.
    """

    simulator: Final[Simulator]
    ll: Final["LinkLayer"]
    node: Final[QNode]
    memory: Final[QuantumMemory]
    ch: Final[QuantumChannel]
    peer: Final[QNode]
    ac: Final[ActiveChannel]

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel, primary_cnt: LinkLayerCounters | None):
        self.simulator = ll.simulator
        self.ll = ll
        self.node = ll.node
        self.memory = ll.memory
        self.ch = ch
        self.peer = ch.find_peer(ll.node)
        self.ac = ActiveChannel(ch, self.peer, primary_cnt)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.node.name}-{self.peer.name}>"

    def send_msg(self, msg: Mapping) -> None:
        """Send a message to the peer node."""
        self.node.send_cpacket(self.peer, ClassicPacket(msg, src=self.node, dest=self.peer))

    @abstractmethod
    def sync_external_enter(self) -> None: ...

    @abstractmethod
    def sync_external_exit(self) -> None: ...

    def handle_release(self, event: QubitReleasedEvent) -> None:
        mq = event.qubit
        ap = self.ac.paths.get(mq.path_id)
        if ap is None or ap.insertion_count == 0:
            self.log_debug("%s ignored reason=no-active-path", event)
            return

        self._release_reuse_qubit(event, ap)

    @abstractmethod
    def _release_reuse_qubit(self, event: QubitReleasedEvent, ap: ActivePath) -> None: ...


@final
class LinkLayerEntanglePri(LinkLayerEntangle):
    """
    Entanglement generation procedure of one quantum channel on the primary node.
    """

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel):
        cnt = LinkLayerCounters(history=ll._cnt_history, rtt=ll.simulator.time(sec=ConstantDelayModel.extract(ch.delay) * 2))
        super().__init__(ll, ch, cnt)

        la = ch.link_arch

        # link_arch.set() may be called multiple times in several situations:
        # - qchannel is activated, deactivated, and re-activated.
        # - qchannel is activated from both sides due to shared paths in opposite directions.
        # Nevertheless, we assume every LinkLayer instance has the same parameters,
        # so that the parameters saved into LinkArch are the same every time.
        la.set(
            time_accuracy=self.simulator.accuracy,
            ch=ch,
            reset_time=self.ll.reset_time,
            tau_0=self.ll.tau_0,
            epr_type=self.node.network.epr_type,
            store_decays=(self.memory.time_decay, self.peer.memory.time_decay),
        )

        epr_tpl = la.make_epr(self.simulator.ts, self.node, self.peer, key=None)

        self.log_debug(
            "CHANNEL_ACTIVATE %s ch=%s link_arch=%s attempt_interval=%s t_notify=%s,%s | template %s",
            self.ac,
            ch.name,
            la.name,
            la.attempt_interval,
            la.d_notify_pri,
            la.d_notify_2nd,
            epr_tpl,
        )

    @property
    def cnt(self) -> "LinkLayerCounters":
        return unwrap_cast(self.ac.cnt)

    @override
    def sync_external_enter(self) -> None:
        # Start reservations.
        self.run_channel()

    @override
    def sync_external_exit(self) -> None:
        # Clear incomplete reservations.
        for ap in self.ac.paths.values():
            ap.oreq_table.clear()

    def run_channel(self, ap: ActivePath | None = None) -> None:
        """
        Start entanglement generation on qubits assigned to a channel.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            ap: If set, only consider qubits allocated to a specified path.
        """
        qubits = self.memory.find(
            (lambda q, _: q.state is QubitState.RAW and q.path_id in self.ac.live_paths)
            if ap is None
            else (lambda q, _: q.state is QubitState.RAW and q.path_id == ap.path_id),
            qchannel=self.ch,
        )
        for q, v in qubits:
            assert q.key is None
            assert v is None
            self.start_reservation(None, q)

    def start_reservation(self, ap: ActivePath | None, mq: MemoryQubit) -> None:
        """
        Start entanglement generation on a qubit assigned to a channel.

        Args:
            ap: ActivePath record, if known.
            mq: The qubit, which must be in RAW state.
        """
        ap = ap or self.ac.paths[mq.path_id]
        assert ap.insertion_count > 0

        # Generate a unique reservation key to represent this entanglement generation protocol execution
        # and the entanglement, which is valid until the entanglement is released.
        mq.key = _AUTOID()
        mq.state = QubitState.ACTIVE

        # Remember the pending reservation in ActivePath record.
        ap.oreq_table[mq.key] = EntangleTask(self.ac, ap, mq)
        self.log_debug(
            "RESERVE_REQ(%s:%s, %s) sent addr=%s peer=%s", self.node.name, mq.path_id, mq.key, mq.addr, self.peer.name
        )

        # Transmit RESERVE_REQ message to the secondary node.
        self.send_msg(ReserveMsg(cmd="RESERVE_REQ", path_id=mq.path_id, key=mq.key))

    def handle_reserve_res(self, req: ReserveParsedMsg) -> None:
        ap = self.ac.paths.get(req.path_id)
        if ap is None or ap.insertion_count == 0:
            self.log_debug("%s ignored reason=no-active-path", req)
            return None

        task = ap.oreq_table.get(req.key, None)
        if task is None:
            self.log_debug("%s ignored reason=no-active-task", req)
            return

        mq = task.mq
        assert mq.key == req.key
        mq.state = QubitState.RESERVED
        self.log_debug("%s processed addr=%s", req, mq.addr)
        self.generate_entanglement(ap, task)

    def generate_entanglement(self, ap: ActivePath, task: EntangleTask) -> None:
        """
        Schedule a successful entanglement attempt using skip-ahead sampling.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            ap: ActivePath record of the path to which the qubit is allocated.
            mq: The memory qubit used for this attempt.
        """
        assert ap.insertion_count > 0
        assert task.k == -1
        la = self.ch.link_arch
        key = task.key

        # Calculate which attempt would succeed (1-based).
        task.k = self.ll._calc_attempt(self.ch)

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
                "EPR_SKIP peer=%s key=%s attempts=%s t_success=%s t_notify=%s,%s reason=beyond-external-phase",
                self.peer.name,
                key,
                task.k,
                task.t_success,
                t_notify_pri,
                t_notify_2nd,
            )
            return

        epr = la.make_epr(task.t_success, self.node, self.peer, key=key)

        # If the network uses ASYNC timing mode or the successful attempt can complete within the current EXTERNAL phase,
        # schedule the EPR arrival events on both nodes.
        self.log_debug(
            "EPR_PREPARE peer=%s key=%s attempts=%s epr=%s t_success=%s t_notify=%s,%s",
            self.peer.name,
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

    @override
    def _release_reuse_qubit(self, event: QubitReleasedEvent, ap: ActivePath) -> None:
        self.log_debug("%s processed", event)
        if event.is_decoh:
            self.ac.cnt.n_decoh += 1
        if self.node.timing.is_async():
            self.start_reservation(ap, event.qubit)


@final
class LinkLayerEntangle2nd(LinkLayerEntangle):
    """
    Entanglement generation procedure of one quantum channel on the secondary node.
    """

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel):
        super().__init__(ll, ch, None)
        self.log_debug("CHANNEL_ACTIVATE %s ch=%s", self.ac, ch.name)

    @override
    def sync_external_enter(self) -> None:
        pass

    @override
    def sync_external_exit(self) -> None:
        # Clear unsatisfied reservation requests.
        for ap in self.ac.paths.values():
            ap.ireq_queue.clear()

    def handle_reserve_req(self, req: ReserveParsedMsg) -> None:
        ap = self.ac.paths.get(req.path_id)
        if ap is None or ap.insertion_count == 0:
            self.log_debug("%s ignored reason=no-active-path", req)
            return None

        if len(ap.ireq_queue) > 0 or not self.try_accept_reservation(ap, req.key):
            # If the queue is non-empty, try_accept_reservation() would not have succeeded.
            # If try_accept_reservation() was unsuccessful, enqueue the request.
            ap.ireq_queue.append(req.key)

    def try_accept_reservation(self, ap: ActivePath, key: str, *, hint: MemoryQubit | None = None) -> bool:
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
                    qchannel=self.ch,  # assigned to the quantum channel
                ),
                (None, None),
            )

        if mq is None:
            return False

        self.log_debug("RESERVE_REQ(%s:%s, %s) accepted addr=%s", self.peer.name, ap.path_id, key, mq.addr)
        mq.state = QubitState.RESERVED
        mq.key = key
        self.send_msg(ReserveMsg(cmd="RESERVE_RES", path_id=ap.path_id, key=key))
        return True

    @override
    def _release_reuse_qubit(self, event: QubitReleasedEvent, ap: ActivePath) -> None:
        self.log_debug("%s processed", event)
        if ap.ireq_queue and self.try_accept_reservation(ap, ap.ireq_queue[0], hint=event.qubit):
            ap.ireq_queue.popleft()
