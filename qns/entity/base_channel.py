from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

from qns.entity.entity import Entity
from qns.entity.node import Node
from qns.models.delay import DelayInput, parseDelay
from qns.simulator import Simulator, Time
from qns.utils import get_rand, log

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

NodeT = TypeVar("NodeT", bound=Node)

if TYPE_CHECKING:
    from qns.entity.node import Node


class BaseChannelInitKwargs(TypedDict, total=False):
    bandwidth: int
    delay: DelayInput
    drop_rate: float
    max_buffer_size: int
    length: float


class BaseChannel(Entity, Generic[NodeT]):
    def __init__(self, name: str, **kwargs: Unpack[BaseChannelInitKwargs]):
        super().__init__(name=name)
        self.node_list: list[NodeT] = []
        self.bandwidth = kwargs.get("bandwidth", 0)
        assert self.bandwidth >= 0
        self.delay_model = parseDelay(kwargs.get("delay", 0))
        self.drop_rate = kwargs.get("drop_rate", 0.0)
        assert 0.0 <= self.drop_rate <= 1.0
        self.max_buffer_size = kwargs.get("max_buffer_size", 0)
        self.length = kwargs.get("length", 0.0)
        assert self.length >= 0.0
        self._next_send_time: Time

    def install(self, simulator: Simulator) -> None:
        """``install`` is called before ``simulator`` runs to initialize or set initial events

        Args:
            simulator (Simulator): the simulator

        """
        super().install(simulator)
        self._next_send_time = simulator.ts

    def _send(self, *, packet_repr: str, packet_len: int, next_hop: NodeT, delay: float) -> tuple[bool, Time]:
        simulator = self.simulator

        if next_hop not in self.node_list:
            raise NextHopNotConnectionException(f"{self}: not connected to {next_hop}")

        if self.bandwidth != 0:
            send_time = max(self._next_send_time, simulator.current_time)

            if self.max_buffer_size != 0 and send_time > simulator.current_time + self.max_buffer_size / self.bandwidth:
                # buffer is overflow
                log.debug(f"{self}: drop {packet_repr} due to overflow")
                return True, Time()

            self._next_send_time = send_time + packet_len / self.bandwidth
        else:
            send_time = simulator.current_time

        # random drop
        if self.drop_rate > 0 and get_rand() < self.drop_rate:
            log.debug(f"{self}: drop {packet_repr} due to drop rate")
            return True, Time()

        # add delay
        recv_time = send_time + (self.delay_model.calculate() + delay)
        return False, recv_time

    def find_peer(self, own: NodeT) -> NodeT:
        """
        Return the node in node_list that is not ``own``.

        Raises:
            ValueError: node_list does not have two nodes, or ``own`` is not one of them.
        """
        if len(self.node_list) != 2:
            raise ValueError(f"{self} does not have exactly 2 nodes")
        if self.node_list[0] == own:
            return self.node_list[1]
        if self.node_list[1] == own:
            return self.node_list[0]
        raise ValueError(f"{self} does not connect to {own}")


class NextHopNotConnectionException(Exception):
    pass
