from typing_extensions import override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import QNode
from mqns.models.epr import WernerStateEntanglement
from mqns.network.proactive.fib import FibEntry
from mqns.network.proactive.mux_buffer_space import MuxSchemeFibBase
from mqns.network.proactive.mux_statistical import MuxSchemeDynamicBase, has_intersect_tmp_path_ids
from mqns.network.proactive.select import MemoryWernerIterator, SelectPath, select_path_random
from mqns.utils import log


class MuxSchemeDynamicEpr(MuxSchemeDynamicBase, MuxSchemeFibBase):
    """
    Dynamic EPR Affection multiplexing scheme.
    """

    def __init__(
        self,
        name="dynamic EPR affection",
        *,
        select_path: SelectPath = select_path_random,
    ):
        super().__init__(name)
        self.select_path = select_path

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        possible_path_ids = self._qubit_is_entangled_0(qubit)
        if not possible_path_ids:  # all paths on the channel have been uninstalled
            return

        # TODO: if paths have different swap policies
        #       -> consider only paths for which this qubit may be eligible ??
        _, epr = self.memory.read(qubit.addr, has=WernerStateEntanglement)

        if epr.tmp_path_ids is None:
            # In principle, a random path_id is chosen for each elementary EPR during EPR generation.
            # The necessary information could be carried in the reservation message.
            # For ease of implementation, this choice is made at either primary or secondary node,
            # whichever receives the EPR notification earlier.
            fib_entries = [self.fib.get(pid) for pid in possible_path_ids]
            fib_entry = self.select_path(fib_entries)
            epr.tmp_path_ids = frozenset([fib_entry.path_id])
        else:
            assert len(epr.tmp_path_ids) == 1
            fib_entry = self.fib.get(next(epr.tmp_path_ids.__iter__()))

        log.debug(f"{self.own}: qubit {qubit} has selected path_id {fib_entry.path_id}")

        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

    @override
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryWernerIterator):
        assert mq0.path_id is None
        possible_path_ids = [fib_entry.path_id]
        return (
            (q, v)
            for (q, v) in input
            if has_intersect_tmp_path_ids(v.tmp_path_ids, possible_path_ids)  # has compatible path_id
        )

    @override
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        assert prev_epr.tmp_path_ids is not None
        assert next_epr.tmp_path_ids is not None
        assert prev_epr.tmp_path_ids == next_epr.tmp_path_ids
        new_epr.tmp_path_ids = prev_epr.tmp_path_ids

    @override
    def su_parallel_has_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is not None
        if su_path_id not in my_new_epr.tmp_path_ids:
            raise Exception(f"{self.own}: Unexpected conflictual parallel swapping")
        return False

    @override
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        assert new_epr.tmp_path_ids is not None
        assert other_epr.tmp_path_ids is not None
        assert new_epr.tmp_path_ids == other_epr.tmp_path_ids
        merged_epr.tmp_path_ids = new_epr.tmp_path_ids
