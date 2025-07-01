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

from typing import TYPE_CHECKING

from qns.entity.cchannel import ClassicPacket
from qns.entity.node import Application, Controller, Node
from qns.network import QuantumNetwork
from qns.simulator import Simulator
from qns.utils import log

if TYPE_CHECKING:
    from qns.network.protocol.proactive_forwarder import InstallPathInstructions, InstallPathMsg

swapping_settings = {
    # disable swapping (for studying isolated links)
    "isolation_1": [0, 0, 0],
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

    def __init__(self, *, swapping: str | list[int], purif: dict[str, int] = {}):
        """
        Args:
            swapping: swapping order to use for the S-D path.
                      If this is a string, it must be a key in `swapping_settings` array.
                      If this is a list of ints, it is the swapping order vector.
            purif: purification settings.
                   Each key identifies a qchannel along the S-D path, written like "S-R1".
                   Each value indicates the number of purification rounds at this hop.

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

        self.purif = purif

    def install(self, node: Node, simulator: Simulator):
        super().install(node, simulator)
        self.own = self.get_node(node_type=Controller)
        self.net = self.own.network

        if len(self.swapping_order) > 0:
            # install the test path on QNodes
            self.install_static_path()

    def install_static_path(self):
        """Install a static path between nodes "S" (source) and "D" (destination) of the network topology.
        It currently supports linear chain topology with a single S-D path.
        It computes a path between S and D and installs instructions with a predefined swapping order.

        This method performs the following:
            - Builds the network route table.
            - Locates the source ("S") and destination ("D") nodes in the network.
            - Computes shortest path between source and destination.
            DijkstraRouteAlgorithm is the default algorithm and uses hop counts at metric.
            - Verifies that the number of nodes in the computed path matches the expected number
            of swapping steps based on the configured swapping strategy.
            - Constructs per-channel memory allocation vectors (m_v) assuming buffer-space multiplexing
            defined by the memory capacity of node S.
            - Sends installation instructions to each node along the path using classical channels.

        Raises:
            RuntimeError - no route from S to D.
            ValueError - computed route length does not match swapping order vector.

        Notes:
            - `self.swapping_order` provides the expected swapping instructions.
            - Multiplexing strategy is hardcoded as "B" (buffer-space).
            - Path instructions are transmitted over classical channels using the `"install_path"` command.
            - This method is primarily intended for test or static configuration scenarios, not dynamic routing.
            - Purification scheme has to be specified (hardcoded) in the function.

        """
        self.net.build_route()

        src = self.net.get_node("S")
        dst = self.net.get_node("D")

        route_result = self.net.query_route(src, dst)
        if len(route_result) == 0:
            raise RuntimeError(f"{self.own}: No route from {src} to {dst}")
        path_nodes = route_result[0][2]
        log.debug(f"{self.own}: Computed path: {path_nodes}")

        route = [n.name for n in path_nodes]
        self.install_path_on_route(route, path_id=0, swap=self.swapping_order, purif=self.purif)

    def install_path_on_route(self, route: list[str], *, path_id: int, swap: list[int], purif: dict[str, int] = {}):
        if len(route) != len(swap) or len(route) == 0:
            raise ValueError("swapping order does not match route length")
        for key in purif.keys():
            tokens = key.split("-")
            if len(tokens) != 2 or tokens[0] not in route or tokens[1] not in route:
                raise ValueError(f"purification instruction {key} does not exist in route")

        m_v: list[int] = []
        src_capacity = self.net.get_node(route[0]).get_memory().capacity
        for i in range(len(route) - 1):
            m_v.append(src_capacity)

        for node_name in route:
            qnode = self.net.get_node(node_name)
            instructions: "InstallPathInstructions" = {
                "route": route,
                "swap": swap,
                "mux": "B",
                "m_v": m_v,
                "purif": purif,
            }
            msg: "InstallPathMsg" = {"cmd": "install_path", "path_id": path_id, "instructions": instructions}

            cchannel = self.own.get_cchannel(qnode)
            cchannel.send(ClassicPacket(msg, src=self.own, dest=qnode), next_hop=qnode)
            log.debug(f"{self.own}: send {msg} to {qnode}")
