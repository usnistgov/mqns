from collections import deque
from typing import TYPE_CHECKING, Final, final, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.ll.entangle import LLEntangleEvent, LLEntangleProc, LLPath, LLPathID, ReserveMsg, ReserveParsedMsg
from mqns.network.ll.event import QubitReleasedEvent
from mqns.simulator import Time
from mqns.utils import unwrap_cast

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer


class LLPath2nd(LLPath):
    ireq_queue: Final[deque[str]]
    """
    Queue of incoming RESERVE_REQ received by this node that has not been fulfilled.
    Element is reservation key.
    """

    def __init__(self, path_id: LLPathID):
        super().__init__(path_id)
        self.ireq_queue = deque()

    @override
    def assert_is_empty(self) -> None:
        assert len(self.ireq_queue) == 0, f"{self.path_id}.ireq_queue = {self.ireq_queue}"


@final
class LLEntangleProc2nd(LLEntangleProc[LLPath2nd]):
    """
    Entanglement generation procedure of one quantum channel on the secondary node.
    """

    def __init__(self, ll: "LinkLayer", ch: QuantumChannel):
        super().__init__(ll, ch)
        self.log_debug("CHANNEL_ACTIVATE")

    @override
    def _path_activate_create(self, path_id: LLPathID) -> LLPath2nd:
        return LLPath2nd(path_id)

    @override
    def _path_deactivate_start_draining(self, lp: LLPath2nd) -> Time:
        # No more reservations would be accepted.
        # Memory qubits for accepted reservations are reset via RESERVE_ABORT handler.
        lp.ireq_queue.clear()
        return self.simulator.tc + self.memory.t_cohere

    @override
    def sync_external_enter(self) -> None:
        pass

    @override
    def sync_external_exit(self) -> None:
        # Clear unsatisfied reservation requests.
        for lp in self.paths_active.values():
            lp.ireq_queue.clear()

    def handle_reserve_req(self, req: ReserveParsedMsg) -> None:
        lp = self.paths_active.get(req.path_id)
        if lp is None:
            self.log_debug("%s ignored reason=no-active-path", req)
            return None

        if len(lp.ireq_queue) > 0 or not self.try_accept_reservation(lp, req.key):
            # If the queue is non-empty, try_accept_reservation() would not have succeeded.
            # If try_accept_reservation() was unsuccessful, enqueue the request.
            lp.ireq_queue.append(req.key)

    def handle_reserve_abort(self, req: ReserveParsedMsg) -> None:
        # Do not check whether the LLPath2nd record is active or draining: the primary node has discarded
        # the qubit when it sends RESERVE_ABORT, even if the path is re-activated later.
        # Do not check whether the key exists in ireq_queue: the path deactivation logic clears ireq_queue.

        # Reset the memory, if present.
        if (mq_tuple := self.memory.read(req.key)) is None:
            # Qubit would not be found if the reservation wasn't accepted prior to path deactivation.
            self.log_debug("%s ignored reason=qubit-not-found", req)
            return
        mq, qm = mq_tuple

        self.log_debug("%s reset mq-addr=%s", req, mq.addr)
        assert qm is None
        assert mq.state in (QubitState.ACTIVE, QubitState.RESERVED)
        mq.state = QubitState.RAW

    def try_accept_reservation(self, lp: LLPath2nd, key: str, *, hint: MemoryQubit | None = None) -> bool:
        """
        Accept a reservation if a qubit is available.

        Args:
            lp: LLPath2nd record of the path to which the qubit is allocated.
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
            assert mq.path_id == lp.path_id
        else:
            mq, _ = next(
                self.memory.find(
                    lambda q, _: (
                        q.state is QubitState.RAW  # currently unoccupied and not part of an active reservation
                        and q.path_id == lp.path_id  # allocated to the path_id, if MuxScheme uses path_id
                    ),
                    qchannel=self.ch,  # assigned to the quantum channel
                ),
                (None, None),
            )

        if mq is None:
            return False

        self.log_debug("RESERVE_REQ(%s, %s) accepted mq-addr=%s", lp.path_id, key, mq.addr)
        mq.state = QubitState.RESERVED
        mq.key = key
        self.send_msg(ReserveMsg(cmd="RESERVE_RES", path_id=lp.path_id, key=key))
        return True

    @override
    def _release_reuse_qubit(self, event: QubitReleasedEvent, lp: LLPath2nd) -> None:
        self.log_debug("%s processed", event)
        if lp.ireq_queue and self.try_accept_reservation(lp, lp.ireq_queue[0], hint=event.qubit):
            lp.ireq_queue.popleft()


@final
class LLEntangle2ndEvent(LLEntangleEvent):
    """
    Event to notify the secondary node about successful elementary entanglement.
    """

    def __init__(self, key: str, epr: Entanglement, *, t: Time):
        super().__init__(t, unwrap_cast(epr.dst), unwrap_cast(epr.src), key, epr)
