from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Literal, override

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw import FibPath, PathInstructions, validate_path_instructions
from mqns.network.fw.select import (
    MemoryEprTuple,
    call_select,
    parse_select,
    select_random,
    select_swap_qubit_newest,
    select_swap_qubit_oldest,
)
from mqns.network.ll import PathActivateEvent, PathDeactivateEvent
from mqns.network.proactive.mux import MuxScheme
from mqns.utils import unwrap, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class MuxSchemeDynamicBase(MuxScheme):
    def __init__(self):
        super().__init__()
        self.qchannel_paths_map: MutableMapping[QuantumChannel, list[int]] = {}
        """
        Mapping from QuantumChannel to path_ids activated on the channel.

        Note: the mapped type is a ``list`` instead of a ``set``, to ensure deterministic simulation.
        """

    @override
    def validate_path_instructions(self, inst: PathInstructions):
        validate_path_instructions(inst)

    @override
    def install_path_adj(self, inst: PathInstructions, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = inst
        paths = self.qchannel_paths_map.setdefault(ch, [])
        assert fp.path_id not in paths
        paths.append(fp.path_id)
        if len(paths) > 1:
            return

        self.simulator.sched(
            PathActivateEvent(
                self.node,
                ch,
                path_id=None,
                t=self.simulator.tc,
                is_primary=dir is PathDirection.R,
            )
        )

    @override
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = dir
        paths = self.qchannel_paths_map[ch]
        paths.remove(fp.path_id)
        if len(paths) > 0:
            return

        del self.qchannel_paths_map[ch]
        self.simulator.sched(
            PathDeactivateEvent(
                self.node,
                ch,
                path_id=None,
                t=self.simulator.tc,
            )
        )

    @override
    def list_qubit_epr_path_ids(self, mq: MemoryQubit) -> list[int]:
        return self.qchannel_paths_map.get(unwrap(mq.qchannel), [])


class MuxSchemeStatistical(MuxSchemeDynamicBase):
    """
    Statistical multiplexing scheme.

    Limitation: across all paths, each node is either a repeater or an end node.
    """

    type SelectSwapQubit = Callable[[list[MemoryEprTuple], "Forwarder", MemoryEprTuple], MemoryEprTuple]

    SelectSwapQubit_random: SelectSwapQubit = select_random
    SelectSwapQubit_oldest: SelectSwapQubit = select_swap_qubit_oldest
    SelectSwapQubit_newest: SelectSwapQubit = select_swap_qubit_newest

    type SelectPath = Callable[[list[int], "Forwarder", Entanglement, Entanglement], int | FibPath]

    SelectPath_random: SelectPath = select_random

    def __init__(
        self,
        *,
        select_swap_qubit: SelectSwapQubit | Literal["random", "oldest", "newest"] | None = None,
        coordinated_decisions=False,
        select_path: SelectPath | Literal["random"] | None = None,
    ):
        """
        Args:
            select_swap_qubit: Function to select a qubit to swap with, default is first.
            coordinated_decisions:
                If True, during a parallel swap, the path_id chosen at one node for selecting swap candidates
                is instantly visible at other nodes. This behavior is physically unrealistic. It is implemented
                for comparison purpose.
                If False (default), during a parallel swap, each node selects swap candidates independently,
                and then discards unusable entanglements due to conflictual swap decisions.
            select_path: Function to select a path (FIB entry) to swap into, default is first.
                This has no effect unless coordinated_decisions is True.
        """
        super().__init__()
        self._select_swap_qubit = parse_select(type(self), "SelectSwapQubit_", select_swap_qubit)
        self.coordinated_decisions = coordinated_decisions
        self._select_path = parse_select(type(self), "SelectPath_", select_path)

    @override
    def validate_path_instructions(self, inst: PathInstructions):
        super().validate_path_instructions(inst)

        # swap sequence must be [1, 0, 0, .., 0, 0, 1]
        s0, *s1, s2 = inst["swap"]
        assert s0 == 1 == s2
        assert all(s == 0 for s in s1)

        # purif scheme must be empty / zeros
        assert all(r == 0 for r in inst["purif"].values())

    @override
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        if self.coordinated_decisions and epr.affectionated_path_id >= 0:
            assert epr.affectionated_path_id in unwrap_cast(mq.epr_path_ids)
            mq.epr_path_ids = [epr.affectionated_path_id]

        neighbor, _ = unwrap_cast(mq.partner)

        def calc_rank_diff(path_id: int):
            fp = self.fib.get_path(path_id)
            _, p_rank = fp.find_index_and_swap_rank(neighbor.name)
            return fp.own_swap_rank - p_rank

        rank_diff = [calc_rank_diff(path_id) for path_id in unwrap_cast(mq.epr_path_ids)]
        assert min(rank_diff) == max(rank_diff)  # failure means one node is both repeater and end node, unsupported
        if rank_diff[0] > 0:
            # Own node has higher rank and cannot initiate swap; qubit stays in ENTANGLED1 state.
            return None

        # Own node has lower/equal rank and can initiate swap.
        mq.state = QubitState.PURIF
        # Without FIB entry, purification scheme cannot be specified, set ELIGIBLE for swapping right away.
        mq.state = QubitState.ELIGIBLE
        return None

    @override
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        mq0_path_ids = set(unwrap_cast(mq0.epr_path_ids))

        candidates = self.memory.find(
            lambda q, v: (
                self.qubits_swappable(mq0, q)  # basic condition met
                and not mq0_path_ids.isdisjoint(unwrap_cast(q.epr_path_ids))  # has overlapping epr_path_ids
                and (
                    not self.coordinated_decisions  # uncoordinated
                    or v.affectionated_path_id < 0  # no existing decision
                    or v.affectionated_path_id in mq0_path_ids  # compatible decisions
                )
            ),
            has=self.epr_type,
        )

        # Find another qubit to swap with.
        # The qubit must have overlapping epr_path_ids so that there would be one or more viable paths.
        # In coordinated_decision mode, the physical path_id (written by parallel swapping) must also match.
        mt1 = call_select(candidates, self._select_swap_qubit, self.fw, (mq0, epr0))
        if mt1 is None:
            return None
        mq1, epr1 = mt1
        assert type(epr1) is self.fw.epr_type

        # Select a FIB entry to guide swap updates initially, but the ForwarderSwapProc could pivot as needed.
        # In coordinated_decision mode, the chosen FIB entry is locked onto EPRs and visible to other nodes.
        selected_path = call_select(
            sorted(mq0_path_ids.intersection(unwrap_cast(mq1.epr_path_ids))),
            self._select_path,
            self.fw,
            epr0,
            epr1,
        )
        assert selected_path is not None, "select_path did not find path"
        fp = selected_path if type(selected_path) is FibPath else self.fib.get_path(selected_path)
        if self.coordinated_decisions:
            epr0.affectionated_path_id = epr1.affectionated_path_id = fp.path_id
        return mq1, fp
