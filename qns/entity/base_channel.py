from typing import Generic, TypedDict, TypeVar

from qns.entity.entity import Entity
from qns.entity.node import NodeT
from qns.models.delay import DelayInput, parseDelay
from qns.simulator import Simulator, Time
from qns.utils import get_rand, log

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

default_light_speed: float = 2e5
"""
Default speed of light in km/s.
Initial value is 200000 km/s, the speed of light in fiber optics.
"""


def set_default_light_speed(light_speed: float) -> None:
    """Change default speed of light in km/s."""
    global default_light_speed
    default_light_speed = light_speed


class BaseChannelInitKwargs(TypedDict, total=False):
    bandwidth: int
    """Bandwidth in bytes/qubits per second. 0 means infinite."""
    max_buffer_size: int
    """
    Maximum buffer size in bytes/qubits. 0 means infinite.
    Packets/photons queued due to bandwidth limitation are placed in the buffer.
    If the buffer is full, the next packet/photon is dropped.
    """
    length: float
    """Link length in kilometers."""
    delay: DelayInput
    """
    Propagation delay in seconds or a DelayModel instance.
    Default is inferred from link length and speed of light, or 0 if link length is unset.
    """
    drop_rate: float
    """Packet/photon loss probability. 0 means never, 1 means always."""


class BaseChannel(Entity, Generic[NodeT]):
    def __init__(self, name: str, **kwargs: Unpack[BaseChannelInitKwargs]):
        super().__init__(name=name)
        self.node_list: list[NodeT] = []
        self._next_send_time: Time

        self.bandwidth = kwargs.get("bandwidth", 0)
        assert self.bandwidth >= 0

        self.max_buffer_size = kwargs.get("max_buffer_size", 0)
        assert self.max_buffer_size >= 0

        self.length = kwargs.get("length", 0.0)
        assert self.length >= 0.0

        self.delay_model = parseDelay(kwargs.get("delay", 0 if self.length == 0 else self.length / default_light_speed))

        self.drop_rate = kwargs.get("drop_rate", 0.0)
        assert 0.0 <= self.drop_rate <= 1.0

    def install(self, simulator: Simulator) -> None:
        """``install`` is called before ``simulator`` runs to initialize or set initial events

        Args:
            simulator (Simulator): the simulator

        """
        super().install(simulator)
        self._next_send_time = simulator.ts

    def _send(self, *, packet_repr: str, packet_len: int, next_hop: NodeT) -> tuple[bool, Time]:
        simulator = self.simulator

        if next_hop not in self.node_list:
            raise NextHopNotConnectionException(f"{self}: not connected to {next_hop}")

        if self.bandwidth != 0:
            send_time = max(self._next_send_time, simulator.tc)

            if self.max_buffer_size != 0 and send_time > simulator.tc + self.max_buffer_size / self.bandwidth:
                # buffer is overflow
                log.debug(f"{self}: drop {packet_repr} due to overflow")
                return True, Time()

            self._next_send_time = send_time + packet_len / self.bandwidth
        else:
            send_time = simulator.tc

        # random drop
        if self.drop_rate > 0 and get_rand() < self.drop_rate:
            log.debug(f"{self}: drop {packet_repr} due to drop rate")
            return True, Time()

        # add delay
        recv_time = send_time + self.delay_model.calculate()
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


ChannelT = TypeVar("ChannelT", bound=BaseChannel)
"""Either ClassicChannel or QuantumChannel."""
