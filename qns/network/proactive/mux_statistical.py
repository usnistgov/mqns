import random

from qns.entity.memory import MemoryQubit, QubitState
from qns.entity.node import QNode
from qns.models.epr import WernerStateEntanglement
from qns.network.proactive.fib import FIBEntry
from qns.network.proactive.message import InstallPathInstructions
from qns.network.proactive.mux import MuxScheme
from qns.utils import log

try:
    from typing import override
except ImportError:
    from typing_extensions import override


def _can_enter_purif(own_name: str, partner_name: str) -> bool:
    """
    Evaluate if a qubit is eligible for purification, in statistical_mux only with limited support.

    - Any entangled qubit at intermediate node is always eligible.
    - Entangled qubit at end-node is eligible only if entangled with another end-node.
    """
    return (
        (own_name.startswith("R"))
        or (own_name.startswith("S") and partner_name.startswith("D"))
        or (own_name.startswith("D") and partner_name.startswith("S"))
    )


def _intersect_tmp_path_ids(epr0: WernerStateEntanglement, epr1: WernerStateEntanglement) -> frozenset[int]:
    assert epr0.tmp_path_ids is not None
    assert epr1.tmp_path_ids is not None
    path_ids = epr0.tmp_path_ids.intersection(epr1.tmp_path_ids)
    if not path_ids:
        raise Exception(f"Cannot select path ID from {epr0.tmp_path_ids} and {epr1.tmp_path_ids}")
    return path_ids


class MuxSchemeDynamicBase(MuxScheme):
    @override
    def validate_path_instructions(self, instructions: InstallPathInstructions):
        assert instructions["mux"] == "S"

    def _qubit_is_entangled_0(self, qubit: MemoryQubit) -> list[int]:
        assert qubit.path_id is None
        if qubit.qchannel is None:
            raise Exception(f"{self.own}: No qubit-qchannel assignment. Not supported.")
        try:
            possible_path_ids = self.fw.qchannel_paths_map[qubit.qchannel.name]
        except KeyError:
            raise Exception(f"{self.own}: qchannel {qubit.qchannel.name} not mapped to any path.")
        return possible_path_ids


class MuxSchemeStatistical(MuxSchemeDynamicBase):
    def __init__(self, name="statistical multiplexing"):
        super().__init__(name)

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        possible_path_ids = self._qubit_is_entangled_0(qubit)
        _, epr = self.memory.get(address=qubit.addr, must=True)
        assert isinstance(epr, WernerStateEntanglement)
        log.debug(f"{self.own}: qubit {qubit}, set possible path IDs = {possible_path_ids}")
        epr.tmp_path_ids = frozenset(possible_path_ids)  # to coordinate decisions along the path
        if _can_enter_purif(self.own.name, neighbor.name):
            self.own.get_qchannel(neighbor)  # ensure qchannel exists
            qubit.state = QubitState.PURIF
            self.fw.qubit_is_purif(qubit)

    @override
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FIBEntry | None) -> None:
        _ = fib_entry
        assert qubit.qchannel is not None

        if not self.own.name.startswith("R"):  # this is an end node
            self.fw.consume_and_release(qubit)
            return

        # this is an intermediate node
        # look for another eligible qubit

        # find qchannels whose qubits may be used with this qubit
        _, epr0 = self.memory.get(address=qubit.addr, must=True)
        assert isinstance(epr0, WernerStateEntanglement)
        assert epr0.tmp_path_ids is not None
        # use path_ids to look for acceptable qchannels for swapping, excluding the qubit's qchannel
        matched_channels = {
            channel
            for channel, path_ids in self.fw.qchannel_paths_map.items()
            if not epr0.tmp_path_ids.isdisjoint(path_ids) and channel != qubit.qchannel.name
        }

        # select qubits based on qchannels only
        res = self.fw._select_eligible_qubit(
            exc_qchannel=qubit.qchannel.name, inc_qchannels=list(matched_channels), tmp_path_id=epr0.tmp_path_ids
        )

        if not res:
            return

        _, epr1 = self.memory.get(address=res.addr, must=True)
        assert isinstance(epr1, WernerStateEntanglement)
        path_ids = _intersect_tmp_path_ids(epr0, epr1)
        fib_entry = self.fib.get_entry(random.choice(list(path_ids)), must=True)  # no need to coordinate across the path
        self.fw.do_swapping(qubit, res, fib_entry, fib_entry)

    @override
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        new_epr.tmp_path_ids = _intersect_tmp_path_ids(prev_epr, next_epr)

    @override
    def su_parallel_avoid_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is not None
        if su_path_id not in my_new_epr.tmp_path_ids:
            log.debug(f"{self.own}: Conflictual parallel swapping in statistical mux -> silently ignore")
            return True
        return False

    @override
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        merged_epr.tmp_path_ids = _intersect_tmp_path_ids(new_epr, other_epr)
