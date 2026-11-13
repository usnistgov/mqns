from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw import FibPath, PathDirection, PathInstructions, validate_path_instructions
from mqns.network.fw.select import (
    MemoryEprTuple,
    call_select,
    parse_select,
    select_random,
    select_swap_qubit_newest,
    select_swap_qubit_oldest,
)
from mqns.network.ll import LLActivateEvent, LLDeactivateEvent
from mqns.network.proactive.mux import MuxScheme
from mqns.utils import unwrap, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class MuxSchemeFibBase(MuxScheme):
    class SelectSwapQubit(Protocol):
        """
        Function to select a preferred swap candidate.

        When qubit ``mq0`` on channel ``ch0`` becomes ELIGIBLE and the forwarder decides to swap it with a qubit from ``ch1``,
        but there are multiple candidate qubits from ``ch1``, this function is called to select a qubit.
        """

        def __call__(
            self, candidates: Sequence[MemoryEprTuple], fw: "Forwarder", mq0: MemoryEprTuple, fp: FibPath, /
        ) -> MemoryEprTuple:
            """
            Args:
                candidates: List of qubits from ``ch1``, guaranteed to have two or more items.
                fw: The forwarder instance.
                mq0: The qubit from ``ch0``.
                fp: FIB path entry.

            Returns:
                One item from ``candidates``.
            """
            ...

    type SelectSwapQubitInput = SelectSwapQubit | Literal["random", "oldest", "newest"]

    SelectSwapQubit_random: SelectSwapQubit = select_random
    """
    Select a random qubit among the candidates, following uniform distribution.
    """
    SelectSwapQubit_oldest: SelectSwapQubit = select_swap_qubit_oldest
    """
    Select the qubit that becomes ELIGIBLE in this forwarder the earliest, i.e. First-In-First-Out (FIFO).
    """
    SelectSwapQubit_newest: SelectSwapQubit = select_swap_qubit_newest
    """
    Select the qubit that becomes ELIGIBLE in this forwarder the latest, i.e. Last-In-First-Out (LIFO).
    """

    def __init__(self, select_swap_qubit: SelectSwapQubitInput):
        super().__init__()
        self._select_swap_qubit = parse_select(type(self), "SelectSwapQubit_", select_swap_qubit)

    def select_swap_candidate(
        self, mt0: MemoryEprTuple, fp: FibPath, predicate: Callable[[MemoryQubit, Entanglement], bool]
    ) -> tuple[MemoryQubit, FibPath] | None:
        candidates = self.memory.find(predicate, has=self.epr_type)
        mt1 = call_select(candidates, self._select_swap_qubit, self.fw, mt0, fp)
        return None if mt1 is None else (mt1[0], fp)


class MuxSchemeBufferSpace(MuxSchemeFibBase):
    """
    Buffer-Space multiplexing scheme.
    """

    def __init__(
        self,
        *,
        select_swap_qubit: MuxSchemeFibBase.SelectSwapQubitInput = "random",
    ):
        """
        Args:
            select_swap_qubit: Function to select a qubit to swap with.
        """
        super().__init__(select_swap_qubit)

    @override
    def validate_path_instructions(self, inst: PathInstructions) -> None:
        validate_path_instructions(inst, bufferspace=True)

    @override
    def install_path_adj(
        self, inst: PathInstructions, fp: FibPath, dir: PathDirection, ch: QuantumChannel, peer: QNode
    ) -> None:
        _ = peer
        assert "bufferspace_mv" in inst
        self.simulator.sched(
            LLActivateEvent(
                self.node,
                ch,
                t=self.simulator.tc,
                gid=fp.path_id,
                q_target=inst["bufferspace_mv"][2 * fp.own_idx + (dir.value - 1)],
                is_primary=self._path_adj_is_primary(inst, fp, dir),
            )
        )

    @override
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel, peer: QNode) -> None:
        _ = dir, peer
        self.simulator.sched(
            LLDeactivateEvent(
                self.node,
                ch,
                t=self.simulator.tc,
                gid=fp.path_id,
            )
        )

    @override
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        _ = epr
        path_id = unwrap_cast(mq.partner).ll_gid
        mq.epr_path_ids = [path_id]
        mq.state = QubitState.PURIF
        return self.fib.get_path(path_id)

    @override
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        return self.select_swap_candidate(
            (mq0, epr0),
            unwrap(fp),
            lambda q, _: (
                self.qubits_swappable(mq0, q)  # basic condition met
                and unwrap_cast(q.epr_path_ids)[0] == unwrap_cast(mq0.epr_path_ids)[0]  # on the same path
            ),
        )
