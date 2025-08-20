from qns.entity.memory import MemoryQubit, QubitState
from qns.entity.node import QNode
from qns.models.epr import WernerStateEntanglement
from qns.network.proactive.fib import FIBEntry, is_isolated_links
from qns.network.proactive.message import InstallPathInstructions
from qns.network.proactive.mux import MuxScheme
from qns.utils import log

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class MuxSchemeFibBase(MuxScheme):
    @override
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FIBEntry | None) -> None:
        assert fib_entry is not None
        assert qubit.qchannel is not None

        if is_isolated_links(fib_entry):  # no swapping in isolated links
            self.fw.consume_and_release(qubit)
            return

        route = fib_entry["path_vector"]
        own_idx = route.index(self.own.name)
        if own_idx in (0, len(route) - 1):  # this is an end node
            self.fw.consume_and_release(qubit)
            return

        # this is an intermediate node
        # look for another eligible qubit

        if qubit.path_id is not None:  # static qubit-path allocation is provided
            possible_path_ids = [fib_entry["path_id"]]
            if not self.fw.isolate_paths:
                # if not isolated paths -> include other paths serving the same request
                possible_path_ids = self.fw.request_paths_map[fib_entry["request_id"]]
                log.debug(f"{self.own}: path ids {possible_path_ids}")

            res = self.fw._select_eligible_qubit(
                exc_qchannel=qubit.qchannel.name, exc_direction=qubit.path_direction, path_id=possible_path_ids
            )
        else:  # dynamic EPR-path allocation
            possible_path_ids = [fib_entry["path_id"]]
            res = self.fw._select_eligible_qubit(exc_qchannel=qubit.qchannel.name, tmp_path_id=possible_path_ids)

        if res:  # do swapping
            self.fw.do_swapping(qubit, res, fib_entry, fib_entry)


class MuxSchemeBufferSpace(MuxSchemeFibBase):
    def __init__(self, name="buffer-space multiplexing"):
        super().__init__(name)

    @override
    def validate_path_instructions(self, instructions: InstallPathInstructions) -> None:
        assert instructions["mux"] == "B"

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        assert qubit.path_id is not None
        fib_entry = self.fib.get_entry(qubit.path_id, must=True)
        qubit.purif_rounds = 0
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

    @override
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        assert prev_epr.tmp_path_ids is None
        assert next_epr.tmp_path_ids is None
        _ = new_epr

    @override
    def su_parallel_avoid_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is None
        _ = su_path_id
        return False

    @override
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        assert new_epr.tmp_path_ids is None
        assert other_epr.tmp_path_ids is None
        _ = merged_epr
