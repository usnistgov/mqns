#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from enum import Enum, auto
from typing import TYPE_CHECKING

from qns.entity.cchannel import ClassicPacket
from qns.entity.node import Application, Controller, Node
from qns.network import QuantumNetwork
from qns.simulator import Simulator
from qns.utils import log

if TYPE_CHECKING:
    from qns.network.protocol.proactive_forwarder import InstallPathInstructions, InstallPathMsg


class QubitAllocationType(Enum):
    MIN_CAPACITY = auto()
    FOLLOW_QCHANNEL = auto()


swapping_settings = {
    # disable swapping (for studying isolated links)
    "no_swap": [0, 0, 0],
    # for 1-repeater
    "swap_1": [1, 0, 1],
    # for 2-repeater
    "swap_2_asap": [1, 0, 0, 1],
    "swap_2_l2r": [2, 0, 1, 2],
    "swap_2_r2l": [2, 1, 0, 2],
    # for 3-repeater
    "swap_3_asap": [1, 0, 0, 0, 1],
    "swap_3_baln": [2, 0, 1, 0, 2],
    "swap_3_l2r": [3, 0, 1, 2, 3],
    "swap_3_r2l": [3, 2, 1, 0, 3],
    "swap_3_vora_uniform": [3, 0, 2, 1, 3],  # equiv. [2,0,1,0,2] ~ baln
    "swap_3_vora_increasing": [3, 0, 1, 2, 3],
    "swap_3_vora_decreasing": [3, 2, 1, 0, 3],
    "swap_3_vora_mid_bottleneck": [3, 1, 2, 0, 3],  # [2,0,1,0,2]  ~ baln
    # for 4-repeater
    "swap_4_asap": [1, 0, 0, 0, 0, 1],
    "swap_4_baln": [3, 0, 1, 0, 2, 3],
    "swap_4_baln2": [3, 2, 0, 1, 0, 3],
    "swap_4_l2r": [4, 0, 1, 2, 3, 4],
    "swap_4_r2l": [4, 3, 2, 1, 0, 4],
    "swap_4_vora_uniform": [4, 0, 3, 1, 2, 4],  # equiv. [3,0,2,0,1,3]
    "swap_4_vora_increasing": [4, 0, 1, 3, 2, 4],  # equiv. [3,0,1,2,0,3]
    "swap_4_vora_decreasing": [4, 3, 1, 2, 0, 4],  # equiv. [3,2,0,1,0,3]
    "swap_4_vora_mid_bottleneck": [4, 0, 2, 3, 1, 4],  # equiv. [3,0,1,2,0,3]
    "swap_4_vora_uniform2": [3, 0, 2, 0, 1, 3],
    "swap_4_vora_increasing2": [3, 0, 1, 2, 0, 3],
    "swap_4_vora_decreasing2": [3, 2, 0, 1, 0, 3],
    "swap_4_vora_mid_bottleneck2": [3, 0, 1, 2, 0, 3],
    # for 5-repeater example
    "swap_5_asap": [1, 0, 0, 0, 0, 0, 1],
    "swap_5_baln": [3, 0, 1, 0, 2, 0, 3],  # need to specify exact doubling  => this is used in the vora paper
    "swap_5_baln2": [3, 0, 2, 0, 1, 0, 3],
    "swap_5_l2r": [5, 0, 1, 2, 3, 4, 5],
    "swap_5_r2l": [5, 4, 3, 2, 1, 0, 5],
    "swap_5_vora_uniform": [5, 0, 3, 1, 4, 2, 5],  # [3,0,1,0,2,0,3]  ~ baln
    "swap_5_vora_increasing": [5, 0, 3, 1, 4, 2, 5],  # [3,0,1,0,2,0,3] ~ baln
    "swap_5_vora_decreasing": [5, 2, 4, 1, 3, 0, 5],  # [3,0,2,0,1,0,3] ~ baln2
    "swap_5_vora_mid_bottleneck": [5, 0, 4, 2, 3, 1, 5],  # [3,0,2,0,1,0,3] ~ baln2
}
"""Predefined swapping orders."""


class ProactiveRoutingControllerApp(Application):
    """
    Centralized control plane app for Proactive Routing.
    Works with Proactive Forwarder on quantum nodes.
    """

    def __init__(
        self,
        *,
        swapping: str | list[int] = [],
        swapping_policy: str | None = None,
        purif: dict[str, int] = {},
        routing_type: str | None = None,
        qubit_allocation: QubitAllocationType = QubitAllocationType.FOLLOW_QCHANNEL,
    ):
        """
        Args:
            swapping: swapping order to use for the S-D path.
                      If this is a string, it must be a key in `swapping_settings` array.
                      If this is a list of ints, it is the swapping order vector.
            swapping_policy: swapping order or policy without specific path length.
                    Accepted values: `l2r`, `r2l`, `baln`, `asap`.
            purif: purification settings.
                   Each key identifies a qchannel along the S-D path, written like "S-R1".
                   Each value indicates the number of purification rounds at this hop.
            routing_type (str): Type of routing to run on the topology. Supported values:
                - SRSP (Single Request Single Path).
                - SRMP_STATIC (Single Request Multiple Paths with qubit-path pre-allocation).
                - SRMP_DYNAMIC (Single Request Multiple Paths without qubit-path pre-allocation).
            qubit_allocation (QubitAllocationType): Type of qubit-path allocation to use with buffer-space mux.
                Supported values:
                - MIN_CAPACITY: uses the lowest qchannel capaciy along the path.
                - FOLLOW_QCHANNEL: uses special value `0` to let nodes use all the qubits assigned to the qchannel
                        connecting to the neighbor.

        To disable automatic installation of a static route from S to D, pass swapping=[].
        """
        super().__init__()
        self.net: QuantumNetwork
        """QN physical topology and classical topology"""
        self.own: Controller
        """controller node running this app"""

        try:
            self.swapping_order = swapping if isinstance(swapping, list) else swapping_settings[swapping]
        except KeyError:
            raise KeyError(f"{self.own}: Swapping {swapping} not configured")

        self.swapping_policy = swapping_policy

        self.purif = purif

        if routing_type and routing_type not in ["SRSP", "SRMP_STATIC", "MRSP_DYNAMIC"]:
            raise Exception(
                f"{self.own}: Routing type {self.routing_type} not supported."
                f"Supported types: ['SRSP', 'SRMP_STATIC', 'MRSP_DYNAMIC']"
            )
        self.routing_type = routing_type

        self.qubit_allocation = qubit_allocation

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=Controller)
        self.net = self.own.network

        # install the test path on QNodes
        if self.routing_type == "SRSP":
            self.do_one_path(0, 0, "S", "D", self.qubit_allocation)
        elif self.routing_type == "SRMP_STATIC":
            self.do_multiple_static_paths()
        elif self.routing_type == "MRSP_DYNAMIC":
            self.do_multirequest_dynamic_paths()

    def do_one_path(self, req_id: int, path_id: int, src: str, dst: str, qubit_allocation: QubitAllocationType | None = None):
        """Install a static path between src and dst of the network topology.
        It currently supports linear chain topology with a single src-dst path.
        It computes a path between src and dst and installs instructions with a predefined swapping order.

        This method performs the following:
            - Builds the network route table.
            - Locates the src and dst nodes in the network and computes shortest path.
            DijkstraRouteAlgorithm is the default algorithm and uses hop counts at metric.
            - Verifies that the number of nodes in the computed path matches the expected number
            of swapping steps based on the configured swapping strategy.
            - Sends installation instructions to each node along the path using classical channels.

        Raises:
            RuntimeError - no route from S to D.
            ValueError - computed route length does not match swapping order vector.

        Notes:
            - `self.swapping_order` provides the expected swapping instructions.
            - Path instructions are transmitted over classical channels using the `"install_path"` command.
            - Purification scheme has to be specified (hardcoded) in the function.
        """
        self.net.build_route()

        src = self.net.get_node(src)
        dst = self.net.get_node(dst)

        route_result = self.net.query_route(src, dst)
        if len(route_result) == 0:
            raise RuntimeError(f"{self.own}: No route from {src} to {dst}")
        path_nodes = route_result[0][2]
        log.debug(f"{self.own}: Computed path: {path_nodes}")

        route = [n.name for n in path_nodes]

        if qubit_allocation is None:
            m_v = None
            mux = "S"
        elif qubit_allocation == QubitAllocationType.MIN_CAPACITY:
            m_v = self.compute_m_v_min_cap(route)
            mux = "B"
        elif qubit_allocation == QubitAllocationType.FOLLOW_QCHANNEL:
            m_v = self.compute_m_v_qchannel(route)
            mux = "B"

        # Use explicit swapping order is given
        if self.swapping_order:
            swap = self.swapping_order
        else:  # define order from policy and path length
            swapping_order = f"swap_{len(path_nodes) - 2}_{self.swapping_policy}"
            if swapping_order not in swapping_settings:
                raise Exception(f"Swapping order {swapping_order} is needed but not found in swapping settings.")
            swap = swapping_settings[swapping_order]

        self.install_path_on_route(route, path_id=path_id, req_id=req_id, m_v=m_v, mux=mux, swap=swap, purif=self.purif)

    def do_multiple_static_paths(self):
        """Install multiple static paths between nodes "S" (source) and "D" (destination) of the network topology.
        It computes k-shortest paths between S and D and installs instructions with a predefined swapping order.
        Paths are not necessarily disjoint; they may share same qchannels.

        This method performs the following:
            - Builds the network route table.
            - Locates the source ("S") and destination ("D") nodes in the network.
            - Computes M shortest paths between source and destination.
            Yen's algorithm is used with hop counts at metric.
            - Computes a swapping sequence for each path based on swapping config (e.g., `l2r` for sequential swapping).
            - Constructs per-channel memory allocation vectors (m_v) assuming buffer-space multiplexing
            following qubits-qchannels allocation defined at topology creation.
            If a qchannel is shared by multiple paths, its qubits are equally divided over the paths.
            - Sends installation instructions to each node along the path using classical channels.

        Raises:
            Exception: If S or D node not found.
            Exception: If the selected swapping configuration is not found for the computed route.

        Notes:
            - Multiplexing strategy is hardcoded as "B" (buffer-space).
            - Path instructions are transmitted over classical channels using the `"install_path"` command.
            - This method is primarily intended for test or static configuration scenarios, not dynamic routing.
            - Purification scheme has to be specified (hardcoded) in the function.
        """
        self.net.build_route()

        src = self.net.get_node("S")
        dst = self.net.get_node("D")

        # Get all shortest paths (M ≥ 1)
        route_result = self.net.query_route(src, dst)  # Expected to be Yen's
        paths = [r[2] for r in route_result]  # list of path_nodes (node objects)

        # Get all quantum channels and initialize usage count
        network_channels = self.net.get_qchannels()
        qchannel_use_count = {ch.name: 0 for ch in network_channels}

        # Count usage of each quantum channel across all paths
        for path_nodes in paths:
            for i in range(len(path_nodes) - 1):
                n1, n2 = path_nodes[i].name, path_nodes[i + 1].name
                ch_name = f"q_{n1},{n2}" if f"q_{n1},{n2}" in qchannel_use_count else f"q_{n2},{n1}"
                qchannel_use_count[ch_name] += 1

        qchannel_names = list(qchannel_use_count.keys())

        # Process each path
        for path_id, path_nodes in enumerate(paths):
            route = [n.name for n in path_nodes]
            log.debug(f"{self.own}: Computed path #{path_id}: {route}")

            # Use explicit swapping order is given
            if self.swapping_order:
                swap = self.swapping_order
            else:  # define order from policy and path length
                swapping_order = f"swap_{len(path_nodes) - 2}_{self.swapping_policy}"
                if swapping_order not in swapping_settings:
                    raise Exception(f"Swapping order {swapping_order} is needed but not found in swapping settings.")
                swap = swapping_settings[swapping_order]

            # Compute buffer-space multiplexing vector as pairs of (qubits_at_node_i, qubits_at_node_i+1)
            # The qubits are divided among all paths that share the qchannel
            m_v = []
            for i in range(len(path_nodes) - 1):
                node_a = path_nodes[i].name
                node_b = path_nodes[i + 1].name

                node_a_obj = self.net.get_node(node_a)
                node_b_obj = self.net.get_node(node_b)

                # From node_i
                ch_name = f"q_{node_a},{node_b}"
                if ch_name not in qchannel_names:
                    raise Exception(f"Qchannel not found: {ch_name}")
                shared = qchannel_use_count[ch_name]

                full_qubits_a = node_a_obj.memory.get_channel_qubits(ch_name)
                qubits_a = len(full_qubits_a) // shared if shared > 0 else len(full_qubits_a)

                # From node_i+1
                full_qubits_b = node_b_obj.memory.get_channel_qubits(ch_name)
                qubits_b = len(full_qubits_b) // shared if shared > 0 else len(full_qubits_b)

                m_v.append((qubits_a, qubits_b))

            # Send install instruction to each node on this path
            self.install_path_on_route(route, path_id=path_id, req_id=0, m_v=m_v, mux="B", swap=swap, purif={})

    def do_multirequest_dynamic_paths(self):
        """Install one path between S1 and D1 and one path between S2 and D2 of the network topology.

        This method uses `do_one_path` to install a path between S1 and D1 (S2 and D2),
        without qubit-path allocation; means dynamic EPR affectation or statistical mux is expected at nodes.
        """
        self.do_one_path(0, 0, "S1", "D1")
        self.do_one_path(1, 1, "S2", "D2")  # keep path_id globally unique

    def compute_m_v_min_cap(self, route: list[str]) -> list[tuple[int, int]]:
        """
        Compute buffer-space multiplexing vector as pairs of (qubits_at_node_i, qubits_at_node_i+1)
        based on minimum memory capacity.
        """
        c = [self.net.get_node(node_name).get_memory().capacity for node_name in route]
        c[0] *= 2
        c[-1] *= 2
        q = min(c) // 2
        return [(q, q) for _ in range(len(route) - 1)]

    def compute_m_v_qchannel(self, route: list[str]) -> list[tuple[int, int]]:
        """
        Compute buffer-space multiplexing vector as pairs of (qubits_at_node_i, qubits_at_node_i+1)
        based on qubit-qchannel assignment defined at topology creation.
        """
        return [(0, 0) for _ in range(len(route) - 1)]

    def install_path_on_route(
        self,
        route: list[str],
        *,
        path_id: int,
        req_id: int,
        mux: str,
        swap: list[int],
        m_v: list[(int, int)] | None = None,
        purif: dict[str, int] = {},
    ):
        """
        Install an explicitly specified path with the given route.

        Args:
            route: a list of node names, in the order they appear in the path.
                   There must a qchannel and a cchannel between adjacent nodes.
            path_id: numeric identifier to uniquely identify this path within the network.
            swap: swap sequence.
                  This list shall have the same length as route.
                  Each integer is the swapping rank of a node, as explained in `find_index_and_swapping_rank`.
            purif: purification instructions.
                   Each key is a segment name consists of two node names concatenated with a hyphen ("-"),
                   where the nodes appear in the same order as in the route but do not have to be adjacent.
                   Each value is an integer of the required rounds of purification at this segment.
                   The default for every segment is zero i.e. no purification is performed.
        """
        if len(route) != len(swap) or len(route) == 0:
            raise ValueError("swapping order does not match route length")
        for segment_name in purif.keys():
            if not check_purif_segment(route, segment_name):
                raise ValueError(f"purif segment {segment_name} does not exist in route")

        for node_name in route:
            qnode = self.net.get_node(node_name)
            instructions: "InstallPathInstructions" = {
                "req_id": req_id,
                "route": route,
                "swap": swap,
                "mux": mux,
                "purif": purif,
            }
            if m_v is not None:
                instructions["m_v"] = m_v
            msg: "InstallPathMsg" = {"cmd": "install_path", "path_id": path_id, "instructions": instructions}

            cchannel = self.own.get_cchannel(qnode)
            cchannel.send(ClassicPacket(msg, src=self.own, dest=qnode), next_hop=qnode)
            log.debug(f"{self.own}: send {msg} to {qnode}")


def check_purif_segment(route: list[str], segment_name: str) -> bool:
    try:
        idx0, idx1 = [route.index(node_name) for node_name in segment_name.split("-")]
        return idx0 < idx1
    except ValueError:
        return False
