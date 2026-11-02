from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Final

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.memory import QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.ll.event import QubitReleasedEvent
from mqns.network.ll.task import ActivePath
from mqns.simulator import Simulator, Time, func_to_event
from mqns.utils import LogSelfMixin

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer


type LLPathID = int | None
"""
Path identifier.

LinkLayer considers ``None`` as a valid path_id (rather than an absent value), because the forwarder's
multiplexing scheme may not need to allocate each qubit to an integer path_id.
"""


class LLEntangleProc(LogSelfMixin):
    """
    Entanglement generation procedure of one quantum channel.
    """

    simulator: Final[Simulator]
    ll: Final["LinkLayer"]
    node: Final[QNode]
    memory: Final[QuantumMemory]
    ch: Final[QuantumChannel]
    peer: Final[QNode]

    paths_active: Final[MutableMapping[LLPathID, ActivePath]]
    """
    Paths that could initiate entanglements.
    """

    paths_draining: Final[MutableMapping[LLPathID, ActivePath]]
    """
    Paths that are pending deletion.
    """

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
        return f"<{type(self).__name__} {self.node.name}-{self.peer.name}>"

    def send_msg(self, msg: Mapping) -> None:
        """Send a message to the peer node."""
        self.node.send_cpacket(self.peer, ClassicPacket(msg, src=self.node, dest=self.peer))

    def path_activate(self, path_id: LLPathID) -> None:
        """Activate a path within the channel."""
        ap = self.paths_draining.pop(path_id, None)
        if ap:
            ap.delete_event.cancel()
        else:
            ap = self._path_activate_create(path_id)

        assert path_id not in self.paths_active, f"{self} already has active path {path_id}"
        self.paths_active[path_id] = ap
        self.log_debug("PATH_ACTIVATE path_id=%s", path_id)

    @abstractmethod
    def _path_activate_create(self, path_id: LLPathID) -> ActivePath: ...

    def path_deactivate(self, path_id: LLPathID) -> None:
        """Deactivate a path within the channel."""
        ap = self.paths_active.pop(path_id, None)
        assert ap is not None, f"{self} does not have active path {path_id}"
        self.paths_draining[path_id] = ap
        t_delete = self._path_deactivate_start_draining(ap)
        self.simulator.sched(delete_event := func_to_event(t_delete, self._path_deactivate_delete, path_id))
        ap.delete_event.set(delete_event)

    @abstractmethod
    def _path_deactivate_start_draining(self, ap: ActivePath) -> Time:
        """
        Start draining the path during deactivation.

        Returns:
            When would draining be complete so the path can be deleted.
        """

    def _path_deactivate_delete(self, path_id: LLPathID) -> None:
        """Delete a path, scheduled when its draining is complete."""
        ap = self.paths_draining.pop(path_id, None)
        assert ap is not None, f"{self} does not have draining path {path_id}"
        self._path_deactivate_verify_empty(ap)
        if len(self.paths_active) + len(self.paths_draining) == 0:
            self.ll.channel_deactivate(self)

    @abstractmethod
    def _path_deactivate_verify_empty(self, ap: ActivePath) -> None:
        """Verify a path has been drained before its deletion."""

    @abstractmethod
    def sync_external_enter(self) -> None: ...

    @abstractmethod
    def sync_external_exit(self) -> None: ...

    def handle_release(self, event: QubitReleasedEvent) -> None:
        mq = event.qubit
        ap = self.paths_active.get(mq.path_id)
        if ap is None:
            self.log_debug("%s ignored reason=no-active-path", event)
            return

        self._release_reuse_qubit(event, ap)

    @abstractmethod
    def _release_reuse_qubit(self, event: QubitReleasedEvent, ap: ActivePath) -> None: ...
