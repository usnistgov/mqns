from collections.abc import Iterable
from typing import TYPE_CHECKING, cast, override

from mqns.entity.memory import PathDirection
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw import (
    FibPath,
    FibRequest,
    Forwarder,
    ForwarderNorthbound,
    PathDeleteMsg,
    PathInsertMsg,
    PathInstructions,
    fw_control_cmd_handler,
)
from mqns.network.proactive.mux import MuxScheme
from mqns.simulator import Time

if TYPE_CHECKING:
    from mqns.network.proactive.forwarder import ProactiveForwarder


def to_fib_erase_delay(t_cohere: Time) -> Time:
    return t_cohere * 4


class ProactiveForwarderNorthbound(ForwarderNorthbound):
    """
    Northbound interface to communicate with ``ProactiveRoutingController``.
    """

    mux: MuxScheme

    @override
    def install(self, fw: Forwarder) -> None:
        super().install(fw)
        self.mux = cast("ProactiveForwarder", fw).mux
        self._fib_erase_delay = to_fib_erase_delay(self.memory.t_cohere)

    @fw_control_cmd_handler("PATH_INSERT")
    def handle_path_insert(self, msg: PathInsertMsg) -> None:
        """Process a PATH_INSERT control command."""
        paths = [self._path_convert(inst) for inst in msg["paths"] if self.node.name in inst["route"]]

        fr = FibRequest(msg["req_id"], [fp for fp, _ in paths], epr_count=msg.get("epr_count", -1))
        self.fib.insert_req(fr)

        # Identify left/right channels, allocate qubits and process LinkLayer changes.
        for fp, inst in paths:
            for dir, ch in self._iter_adjacency(fp):
                self.mux.install_path_adj(inst, fp, dir, ch)

    def _path_convert(self, inst: PathInstructions) -> tuple[FibPath, PathInstructions]:
        self.mux.validate_path_instructions(inst)
        route = inst["route"]

        # Insert FIB entry.
        if "swap_cutoff" in inst:
            swap_cutoff = [None if t < 0 else self.simulator.time(slot=t) for t in inst["swap_cutoff"]]
        else:
            swap_cutoff: list[Time | None] = [None] * (2 * (len(route) - 2))
        return FibPath(
            path_id=inst["path_id"],
            route=route,
            own_idx=route.index(self.node.name),
            swap=inst["swap"],
            swap_cutoff=swap_cutoff,
            purif=inst["purif"],
        ), inst

    @fw_control_cmd_handler("PATH_DELETE")
    def handle_path_delete(self, msg: PathDeleteMsg) -> None:
        """Process a PATH_DELETE control command."""
        fr = self.fib.delete_req(msg["req_id"], delay=self._fib_erase_delay)

        # Identify left/right channels, deallocate qubits and process LinkLayer changes.
        for fp in fr.paths:
            for dir, ch in self._iter_adjacency(fp):
                self.mux.uninstall_path_adj(fp, dir, ch)

    def _iter_adjacency(self, fp: FibPath) -> Iterable[tuple[PathDirection, QuantumChannel]]:
        """Iterate over quantum channels connected to adjacent nodes."""
        for ch_dir, route_off in (PathDirection.L, -1), (PathDirection.R, 1):
            neigh_idx = fp.own_idx + route_off
            if neigh_idx in (-1, len(fp.route)):
                continue
            neigh = self.network.get_node(fp.route[neigh_idx])
            ch = self.node.get_qchannel(neigh)
            yield ch_dir, ch
