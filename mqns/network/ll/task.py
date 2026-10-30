from collections import deque
from typing import TYPE_CHECKING, Final, final, override

from mqns.entity.memory import MemoryQubit
from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.simulator import Event, EventHandleSlot, Time
from mqns.utils import unwrap, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.entangle import LinkLayerEntangle


class ActivePath:
    """
    LinkLayer data structure related to an active path in an active quantum channel.
    """

    path_id: Final[int | None]
    """
    Path identifier.

    LinkLayer considers ``None`` as a valid path_id (rather than an absent value), because the forwarder's
    multiplexing scheme may not need to allocate each qubit to an integer path_id.
    """

    deleting = False
    """
    Whether the path is pending deletion.
    """

    delete_event: EventHandleSlot
    """
    Delayed deletion event, scheduled upon ``insertion_count`` reaches zero.
    """

    oreq_table: dict[str, "EntangleTask"]
    """
    Table of outgoing RESERVE_REQ sent by this node for which the EPR has not arrived.
    This field only exists on the primary node of the channel.
    Key is reservation key.
    Value is EntangleTask record.
    """

    ireq_queue: deque[str]
    """
    Queue of incoming RESERVE_REQ received by this node that has not been fulfilled.
    This field only exists on the secondary node of the channel.
    Element is reservation key.
    """

    def __init__(self, path_id: int | None, is_primary: bool):
        self.path_id = path_id
        self.delete_event = EventHandleSlot()
        if is_primary:
            self.oreq_table = {}
        else:
            self.ireq_queue = deque()

    def __repr__(self) -> str:
        return f"ActivePath({self.path_id})"


class EntangleTask:
    ac: "LinkLayerEntangle"
    ap: ActivePath
    mq: MemoryQubit
    k: int = -1
    """
    Which attempt shall success, starting with 1.
    """
    t_reserve: Time = Time.SENTINEL
    """
    Time point when the primary node receives the RESERVE_RES message and the first attempt begins.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    t_success: Time = Time.SENTINEL
    """
    Time point when the successful attempt begins.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    t_finish: Time = Time.SENTINEL
    """
    Time point when the last notify event occurs.
    ``Time.SENTINEL`` means the reservation has not been accepted.
    """
    notify_pri_event: "EntanglePriEvent|None" = None
    """
    Event to notify primary, cancelable before ``t_success``.
    """
    notify_2nd_event: "Entangle2ndEvent|None" = None
    """
    Event to notify secondary, cancelable before ``t_success``.
    """

    def __init__(self, ac: "LinkLayerEntangle", ap: ActivePath, mq: MemoryQubit):
        self.ac = ac
        self.ap = ap
        self.mq = mq

    @property
    def key(self) -> str:
        return unwrap(self.mq.key)

    def __repr__(self) -> str:
        return f"EntangleTask({self.key}, k={self.k}, t_reserve={self.t_reserve})"


class EntangleEvent(Event):
    def __init__(self, t: Time, node: QNode, partner: QNode, key: str, epr: Entanglement):
        super().__init__(t, f"{node.name}, partner={partner.name}, key={key}, epr={epr.name}")
        self.node = node
        self.partner = partner
        self.key = key
        self.epr = epr

    @override
    def invoke(self) -> None:
        self.node.handle(self)


@final
class EntanglePriEvent(EntangleEvent):
    """
    Event to notify the primary node about successful elementary entanglement.
    """

    def __init__(
        self,
        key: str,
        epr: Entanglement,
        task: EntangleTask,
        *,
        t: Time,
    ):
        super().__init__(t, unwrap_cast(epr.src), unwrap_cast(epr.dst), key, epr)
        self.task = task


@final
class Entangle2ndEvent(EntangleEvent):
    """
    Event to notify the secondary node about successful elementary entanglement.
    """

    def __init__(self, key: str, epr: Entanglement, *, t: Time):
        super().__init__(t, unwrap_cast(epr.dst), unwrap_cast(epr.src), key, epr)
