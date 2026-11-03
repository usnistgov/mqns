from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypedDict, override

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.memory import QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.ll.event import QubitReleasedEvent
from mqns.simulator import Event, EventHandleSlot, Simulator, Time, func_to_event
from mqns.utils import LogSelfMixin, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer


type LLPathID = int | None
"""
Path identifier.

LinkLayer considers ``None`` as a valid path_id (rather than an absent value), because the forwarder's
multiplexing scheme may not need to allocate each qubit to an integer path_id.
"""


class ReserveMsg(TypedDict):
    """
    Message between LinkLayers to request and accept qubit reservation for entanglement generation.
    """

    cmd: Literal["RESERVE_REQ", "RESERVE_ABORT", "RESERVE_RES"]
    """
    * RESERVE_REQ -- pri-to-2nd, request entanglement generation.
    * RESERVE_ABORT -- pri-to-2nd, cancel the reservation and abort entanglement generation.
    * RESERVE_RES -- 2nd-to-pri, accept entanglement generation.
    """
    path_id: LLPathID
    """
    Path identifier assigned to the generated entanglement.
    """
    key: str
    """
    Reservation key that uniquely identifies this reservation and the resulting entanglement.
    """


class ReserveParsedMsg(NamedTuple):
    cmd: str
    path_id: LLPathID
    key: str

    @staticmethod
    def new(msg: ReserveMsg) -> "ReserveParsedMsg":
        return ReserveParsedMsg(cmd=msg["cmd"], path_id=msg["path_id"], key=msg["key"])

    def __repr__(self) -> str:
        return f"{self.cmd}({self.path_id}, {self.key})"


class LLPath(ABC):
    path_id: Final[LLPathID]
    """Path identifier."""

    delete_event: Final[EventHandleSlot]
    """Delayed deletion event."""

    def __init__(self, path_id: LLPathID):
        self.path_id = path_id
        self.delete_event = EventHandleSlot()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path_id})"

    @abstractmethod
    def assert_is_empty(self) -> None:
        """Verify that the LLPath record is empty and can be deleted."""


class LLEntangleProc[LP: LLPath](LogSelfMixin):
    """
    Entanglement generation procedure of one quantum channel.
    """

    simulator: Final[Simulator]
    ll: Final["LinkLayer"]
    node: Final[QNode]
    memory: Final[QuantumMemory]
    ch: Final[QuantumChannel]
    peer: Final[QNode]

    paths_active: Final[MutableMapping[LLPathID, LP]]
    """
    Paths that could initiate entanglements.
    """

    paths_draining: Final[MutableMapping[LLPathID, LP]]
    """
    Paths that are pending deletion.
    """

    _repr = ""
    """``__repr__`` return value, should be assigned by subclass constructor."""

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel):
        self.simulator = ll.simulator
        self.ll = ll
        self.node = ll.node
        self.memory = ll.memory
        self.ch = ch
        self.peer = ch.find_peer(ll.node)
        self.paths_active = {}
        self.paths_draining = {}

    def __repr__(self) -> str:
        return self._repr

    def send_msg(self, msg: Mapping) -> None:
        """Send a message to the peer node."""
        self.node.send_cpacket(self.peer, ClassicPacket(msg, src=self.node, dest=self.peer))

    def path_activate(self, path_id: LLPathID) -> None:
        """Activate a path within the channel."""
        lp = self.paths_draining.pop(path_id, None)
        if lp:
            lp.delete_event.cancel()
        else:
            lp = self._path_activate_create(path_id)

        if path_id in self.paths_active:
            raise ValueError(f"{self} already has active path {path_id}")
        self.paths_active[path_id] = lp
        self.log_debug("%s PATH_ACTIVATE", lp)

    @abstractmethod
    def _path_activate_create(self, path_id: LLPathID) -> LP: ...

    def path_deactivate(self, path_id: LLPathID) -> None:
        """Deactivate a path within the channel."""
        try:
            lp = self.paths_active.pop(path_id)
        except LookupError:
            raise ValueError(f"{self} does not have active path {path_id}") from None
        self.paths_draining[path_id] = lp

        t_delete = self._path_deactivate_start_draining(lp)
        self.simulator.sched(delete_event := func_to_event(t_delete, self._path_deactivate_delete, path_id))
        lp.delete_event.set(delete_event)
        self.log_debug("%s PATH_DEACTIVATE t_delete=%s", lp, t_delete)

    @abstractmethod
    def _path_deactivate_start_draining(self, lp: LP) -> Time:
        """
        Start draining the path during deactivation.

        Returns:
            When would draining be complete so the path can be deleted.
        """

    def _path_deactivate_delete(self, path_id: LLPathID) -> None:
        """Delete a path, scheduled when its draining is complete."""
        try:
            lp = self.paths_draining.pop(path_id)
        except LookupError:
            raise LookupError(f"{self} does not have draining path {path_id}") from None

        lp.assert_is_empty()
        if len(self.paths_active) + len(self.paths_draining) == 0:
            self.ll.channel_deactivate(self)

    @abstractmethod
    def sync_external_enter(self) -> None: ...

    @abstractmethod
    def sync_external_exit(self) -> None: ...

    def handle_release(self, event: QubitReleasedEvent) -> None:
        mq = event.qubit
        lp = self.paths_active.get(mq.path_id)
        if lp is None:
            self.log_debug("%s ignored reason=no-active-path", event)
            return

        self._release_reuse_qubit(event, lp)

    @abstractmethod
    def _release_reuse_qubit(self, event: QubitReleasedEvent, lp: LP) -> None: ...


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
