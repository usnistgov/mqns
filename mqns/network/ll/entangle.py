from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.memory import QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.ll.counter import LinkLayerCounters
from mqns.network.ll.event import QubitReleasedEvent
from mqns.network.ll.task import ActiveChannel, ActivePath
from mqns.simulator import Simulator
from mqns.utils import LogSelfMixin

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer


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
