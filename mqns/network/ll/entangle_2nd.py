from collections import deque
from typing import TYPE_CHECKING, Final, final, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.ll.entangle import LLEntangleProc, LLGroup, LLNotifyEvent, ReserveMsg
from mqns.network.ll.event import QubitReleasedEvent
from mqns.simulator import Time
from mqns.utils import unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer


@final
class LLEntangleProc2nd(LLEntangleProc["LLGroup2nd"]):
    """
    Entanglement generation procedure of one quantum channel on the secondary node.
    """

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel):
        super().__init__(ll, ch, ch_repr_attrs=("peer", "node"))
        self.log_debug("CHANNEL_ACTIVATE")

    @override
    def _group_create(self, gid: int) -> "LLGroup2nd":
        return LLGroup2nd(self, gid)


class LLGroup2nd(LLGroup[LLEntangleProc2nd]):
    ireq_queue: Final[deque[str]]
    """
    Queue of incoming RESERVE_REQ received by this node that has not been fulfilled.
    Element is reservation key.
    """

    def __init__(self, proc: LLEntangleProc2nd, gid: int):
        super().__init__(proc, gid)
        self.ireq_queue = deque()

    @override
    def drain(self) -> Time:
        # No more reservations would be accepted.
        # Memory qubits for accepted reservations are reset via RESERVE_ABORT handler.
        self.ireq_queue.clear()
        return self._drain_time(self.proc.simulator.tc)

    @override
    def assert_is_drained(self) -> None:
        assert len(self.ireq_queue) == 0, f"{self.gid}.ireq_queue = {self.ireq_queue}"

    @override
    def sync_external_enter(self) -> None:
        pass

    @override
    def sync_external_exit(self) -> None:
        # Clear unsatisfied reservation requests.
        self.ireq_queue.clear()

    def handle_reserve_req(self, req: ReserveMsg) -> None:
        if len(self.ireq_queue) > 0 or not self.try_accept_reservation(req.key):
            # If the queue is non-empty, try_accept_reservation() would not have succeeded.
            # If try_accept_reservation() was unsuccessful, enqueue the request.
            self.ireq_queue.append(req.key)

    def handle_reserve_abort(self, req: ReserveMsg) -> None:
        # Do not check whether this group is active or draining: the primary node has discarded
        # the qubit when it sends RESERVE_ABORT, even if the group is re-activated later.
        # Do not check whether the key exists in ireq_queue: the path deactivation logic clears ireq_queue.
        proc = self.proc

        # Reset the memory, if present.
        if (mq_tuple := proc.node.memory.read(req.key)) is None:
            # Qubit would not be found if the reservation wasn't accepted prior to path deactivation.
            self.log_debug("%s ignored reason=qubit-not-found", req)
            return
        mq, qm = mq_tuple

        self.log_debug("%s reset mq-addr=%s", req, mq.addr)
        assert qm is None
        assert mq.state in (QubitState.ACTIVE, QubitState.RESERVED)
        mq.state = QubitState.RAW

    def try_accept_reservation(self, key: str, *, hint: MemoryQubit | None = None) -> bool:
        """
        Accept a reservation if a qubit is available.

        Args:
            key: Reservation key from a received RESERVE_RES message.
            hint: If provided, it will be used, bypassing the search; it must be in RAW state.

        Returns:

            * True if the request is accepted and ``RESERVE_RES`` is sent.
              Caller is responsible for dequeuing the request if it was in a queue.
            * False if the request is not accepted.
              Caller is responsible for enqueuing the request.
        """
        if hint:
            mq = hint
            assert mq.state is QubitState.RAW
            assert mq.addr in self.qubits
        else:
            mq = next((mq for mq in self.qubits.values() if mq.state is QubitState.RAW), None)

        if mq is None:
            return False

        proc = self.proc
        self.log_debug("RESERVE_ACCEPT key=%s mq-addr=%s", key, mq.addr)
        mq.state = QubitState.RESERVED
        mq.key = key
        proc.send_msg(ReserveMsg("RESERVE_RES", self.gid, key))
        return True

    @override
    def handle_release(self, event: QubitReleasedEvent) -> None:
        self.log_debug("%s processed", event)
        if self.ireq_queue and self.try_accept_reservation(self.ireq_queue[0], hint=event.qubit):
            self.ireq_queue.popleft()


@final
class LLNotify2ndEvent(LLNotifyEvent):
    """
    Event to notify the secondary node about successful elementary entanglement.
    """

    def __init__(self, key: str, epr: Entanglement, *, t: Time):
        super().__init__(t, unwrap_cast(epr.dst), unwrap_cast(epr.src), key, epr)
