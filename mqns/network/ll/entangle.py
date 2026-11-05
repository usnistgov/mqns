from abc import abstractmethod
from collections.abc import Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, override

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.ll.event import QubitReleasedEvent
from mqns.simulator import Event, EventHandleSlot, Simulator, Time, func_to_event
from mqns.utils import LogSelfMixin, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer


class ReserveMsg(NamedTuple):
    """
    Message between LinkLayers to request and accept qubit reservation for entanglement generation.
    """

    cmd: Literal["RESERVE_REQ", "RESERVE_ABORT", "RESERVE_RES"]
    """
    * RESERVE_REQ -- pri-to-2nd, request entanglement generation.
    * RESERVE_ABORT -- pri-to-2nd, cancel the reservation and abort entanglement generation.
    * RESERVE_RES -- 2nd-to-pri, accept entanglement generation.
    """
    gid: int
    """
    Group identifier.
    """
    key: str
    """
    Reservation key that uniquely identifies this reservation and the resulting entanglement.
    """

    def __repr__(self) -> str:
        return f"{self.cmd}({self.gid}, {self.key})"


class LLEntangleProc[G: LLGroup[Any]](LogSelfMixin):
    """
    Entanglement generation procedure of one quantum channel.
    """

    simulator: Final[Simulator]
    ll: Final["LinkLayer"]
    ch: Final[QuantumChannel]
    node: Final[QNode]
    peer: Final[QNode]
    _ch_repr: Final[str]  # pri-2nd

    groups_active: Final[MutableMapping[int, G]]
    """
    Groups that could initiate entanglements.
    """

    groups_draining: Final[MutableMapping[int, G]]
    """
    Groups that are pending deletion.
    """

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel, *, ch_repr_attrs: tuple[Literal["node", "peer"], ...]):
        self.simulator = ll.simulator
        self.ll = ll
        self.ch = ch
        self.node = ll.node
        self.peer = ch.find_peer(ll.node)
        self._ch_repr = "-".join(getattr(self, attr).name for attr in ch_repr_attrs)
        self.groups_active = {}
        self.groups_draining = {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self._ch_repr}>"

    def send_msg(self, msg: ReserveMsg) -> None:
        """Send a message to the peer node."""
        self.node.send_cpacket(self.peer, ClassicPacket(msg._asdict(), src=self.node, dest=self.peer))

    def group_activate(self, gid: int, qubits: Iterable[MemoryQubit]) -> None:
        """Activate a group within the channel."""
        grp = self.groups_draining.pop(gid, None)
        if grp:
            grp.delete_event.cancel()
        else:
            grp = self._group_create(gid)
            for mq in qubits:
                grp.qubits[mq.addr] = mq
                self.ll.group_by_mq[mq.addr] = grp

        if gid in self.groups_active:
            raise ValueError(f"{self} already has active group {gid}")
        self.groups_active[gid] = grp
        grp.log_debug("GROUP_ACTIVATE qubits=%s", list(grp.qubits.keys()))
        if self.node.timing.is_async():
            grp.sync_external_enter()

    @abstractmethod
    def _group_create(self, gid: int) -> G: ...

    def group_deactivate(self, gid: int) -> None:
        """Deactivate a group within the channel."""
        try:
            grp = self.groups_active.pop(gid)
        except LookupError:
            raise ValueError(f"{self} does not have active group {gid}") from None
        self.groups_draining[gid] = grp

        t_delete = grp.drain()
        self.simulator.sched(delete_event := func_to_event(t_delete, self._group_delete, gid))
        grp.delete_event.set(delete_event)
        grp.log_debug("GROUP_DEACTIVATE t_delete=%s", t_delete)

    def _group_delete(self, gid: int) -> None:
        """Delete a path, scheduled when its draining is complete."""
        try:
            grp = self.groups_draining.pop(gid)
        except LookupError:
            raise LookupError(f"{self} does not have draining path {gid}") from None

        grp.delete_event.cancel()
        for mq in grp.qubits.values():
            if self.ll.group_by_mq[mq.addr] is grp:
                del self.ll.group_by_mq[mq.addr]

        grp.assert_is_drained()
        if len(self.groups_active) + len(self.groups_draining) == 0:
            self.ll.channel_deactivate(self)


class LLGroup[P: LLEntangleProc[Any]](LogSelfMixin):
    """
    A group of memory qubits allocated to generate elementary entanglements that serve a single purpose.
    """

    proc: Final[P]
    """Channel entangle procedure."""

    gid: Final[int]
    """Group identifier."""

    delete_event: Final[EventHandleSlot]
    """Delayed deletion event."""

    qubits: Final[MutableMapping[int, MemoryQubit]]

    def __init__(self, proc: P, gid: int):
        self.proc = proc
        self.gid = gid
        self.delete_event = EventHandleSlot()
        self.qubits = {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.proc._ch_repr}, {self.gid}>"

    @abstractmethod
    def drain(self) -> Time:
        """
        Start draining the group during deactivation.

        Returns:
            When would draining be complete so the group can be deleted.
        """

    def _drain_time(self, since: Time) -> Time:
        # Drain duration is currently set to t_cohere.
        # This could be shortened as the procedure improves.
        return since + self.proc.node.memory.t_cohere

    @abstractmethod
    def assert_is_drained(self) -> None:
        """Verify that the group has been drained and can be deleted."""

    @abstractmethod
    def sync_external_enter(self) -> None:
        """
        Invoked in SYNC timing upon entering EXTERNAL phase,
        and in ASYNC timing when the group is added.
        """

    @abstractmethod
    def sync_external_exit(self) -> None:
        """
        Invoked in SYNC timing upon exiting EXTERNAL phase.
        """

    @abstractmethod
    def handle_release(self, event: QubitReleasedEvent) -> None: ...


class LLNotifyEvent(Event):
    def __init__(self, t: Time, node: QNode, peer: QNode, key: str, epr: Entanglement):
        assert epr.decohere_time > t
        super().__init__(t, f"{unwrap_cast(epr.src).name}-{unwrap_cast(epr.dst).name}, key={key}, epr={epr.name}")
        self.node = node
        self.peer = peer
        self.key = key
        self.epr = epr

    @override
    def invoke(self) -> None:
        self.node.handle(self)
