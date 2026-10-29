from typing import TYPE_CHECKING, final, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.qchannel import QuantumChannel
from mqns.network.ll.entangle import LLEntangleProc
from mqns.network.ll.event import QubitReleasedEvent
from mqns.network.ll.message import ReserveMsg, ReserveParsedMsg
from mqns.network.ll.task import ActivePath

if TYPE_CHECKING:
    from mqns.network.ll.linklayer import LinkLayer


@final
class LLEntangleProc2nd(LLEntangleProc):
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
