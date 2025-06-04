#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/QNLab-USTC/SimQN),
#    which is licensed under the GNU General Public License v3.0.
#
#    The original SimQN header is included below.


#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
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

from qns.entity import ClassicChannel, Controller, QNode, QuantumChannel, QuantumMemory
from qns.network.requests import Request
from qns.network.route import DijkstraRouteAlgorithm, RouteImpl
from qns.network.topology import ClassicTopology, Topology
from qns.simulator import Simulator, Time, func_to_event
from qns.utils import log


class TimingModeEnum(Enum):
    ASYNC = auto()
    LSYNC = auto()
    SYNC = auto()

class SignalTypeEnum(Enum):
    EXTERNAL_START = auto()     # used by LSYNC to signal new slot
    INTERNAL = auto()           # used by SYNC to set the phase
    EXTERNAL = auto()           # used by SYNC to set the phase
    ROUTING = auto()            # used by SYNC to set the phase
    APP = auto()                # used by SYNC to set the phase

class QuantumNetwork:
    """QuantumNetwork includes quantum nodes, quantum and classical channels, arranged in a given topology
    """

    def __init__(self, *,
                 topo: Topology|None = None, route: RouteImpl|None = None,
                 classic_topo: ClassicTopology|None = None,
                 name: str|None = None,
                 timing_mode: TimingModeEnum = TimingModeEnum.ASYNC,
                 t_slot: float = 0, t_ext: float = 0, t_int: float = 0):
        """Args:
        topo: a `Topology` class.
        route: the routing implement. If route is None, the dijkstra algorithm will be used.
        classic_topo (ClassicTopo): a `ClassicTopo` enum class.
        name: name of the network.

        """
        self.timing_mode = timing_mode
        self.t_slot = t_slot            # for LSYNC
        self.t_ext = t_ext              # for SYNC
        self.t_int = t_int              # for SYNC

        self.name = name
        self.controller: Controller|None = None

        if topo is None:
            self.nodes: list[QNode] = []
            self.qchannels: list[QuantumChannel] = []
            self.cchannels: list[ClassicChannel] = []
        else:
            self.nodes, self.qchannels = topo.build()
            if classic_topo is not None:
                self.cchannels = topo.add_cchannels(classic_topo=classic_topo, nl=self.nodes, ll=self.qchannels)
            else:
                self.cchannels = topo.add_cchannels()

            for n in self.nodes:
                n.add_network(self)

            # set network controller if centralized routing
            if topo.controller:
                self.controller = topo.controller
                self.controller.add_network(self)

        # set quantum routing algorithm
        if route is None:
            self.route: RouteImpl = DijkstraRouteAlgorithm()
        else:
            self.route: RouteImpl = route

        self.requests: list[Request] = []


    def install(self, s: Simulator):
        """Install all nodes (including channels, memories and applications) in this network

        Args:
            simulator (qns.simulator.simulator.Simulator): the simulator

        """
        self._simulator = s

        for n in self.nodes:
            n.install(s)
        if self.controller:
            self.controller.install(s)

        if self.timing_mode == TimingModeEnum.LSYNC and self.t_slot > 0:
            event = func_to_event(self._simulator.ts, self.send_sync_signal, by=self)
            self._simulator.add_event(event)
        elif self.timing_mode == TimingModeEnum.SYNC and self.t_ext > 0 and self.t_int > 0:
            event = func_to_event(self._simulator.ts, self.send_ext_signal, by=self)
            self._simulator.add_event(event)

    def send_sync_signal(self):
        # insert the next send_sync_signal
        t_next = self._simulator.tc + Time(sec=self.t_slot)
        next_event = func_to_event(t_next, self.send_sync_signal, by=self)
        self._simulator.add_event(next_event)

        log.debug("TIME_SYNC: signal EXTERNAL_START")
        # TODO: add controller
        for node in self.nodes:
            node.handle_sync_signal(SignalTypeEnum.EXTERNAL_START)

    def send_ext_signal(self):
        # insert the INT phase after t_ext
        t_int = self._simulator.tc + Time(sec=self.t_ext)
        int_event = func_to_event(t_int, self.send_int_signal, by=self)
        self._simulator.add_event(int_event)

        log.debug("TIME_SYNC: signal EXTERNAL phase")
        # TODO: add controller
        for node in self.nodes:
            node.handle_sync_signal(SignalTypeEnum.EXTERNAL)

    def send_int_signal(self):
        # insert the EXT phase after t_int
        t_ext = self._simulator.tc + Time(sec=self.t_int)
        ext_event = func_to_event(t_ext, self.send_ext_signal, by=self)
        self._simulator.add_event(ext_event)

        log.debug("TIME_SYNC: signal INTERNAL phase")
        # TODO: add controller
        for node in self.nodes:
            node.handle_sync_signal(SignalTypeEnum.INTERNAL)


    def get_nodes(self):
        return self.nodes

    def get_cchannels(self):
        return self.cchannels

    def get_qchannels(self):
        return self.qchannels

    def add_node(self, node: QNode):
        """Add a QNode into this network.

        Args:
            node (qns.entity.node.node.QNode): the inserting node

        """
        self.nodes.append(node)
        node.add_network(self)

    def get_node(self, name: str) -> QNode:
        """Get the QNode by its name

        Args:
            name (str): its name
        Returns:
            the QNode

        """
        for n in self.nodes:
            if n.name == name:
                return n
        raise IndexError(f"node {name} does not exist")

    def set_controller(self, controller: Controller):
        """Set the controller of this network.

        Args:
            node (qns.entity.node.node.Controller): the controller node

        """
        self.controller = controller
        controller.add_network(self)

    def get_controller(self) -> Controller:
        """Get the Controller of this network

        Args:
            name (str): its name
        Returns:
            the Controller

        """
        if self.controller is None:
            raise IndexError("network does not have a controller")
        return self.controller

    def add_qchannel(self, qchannel: QuantumChannel):
        """Add a QuantumChannel into this network.

        Args:
            qchannel (qns.entity.qchannel.qchannel.QuantumChannel): the inserting QuantumChannel

        """
        self.qchannels.append(qchannel)

    def get_qchannel(self, name: str) -> QuantumChannel:
        """Get the QuantumChannel by its name

        Args:
            name (str): its name
        Returns:
            the QuantumChannel

        """
        for n in self.qchannels:
            if n.name == name:
                return n
        raise IndexError(f"qchannel {name} does not exist")

    def add_cchannel(self, cchannel: ClassicChannel):
        """Add a ClassicChannel into this network.

        Args:
            cchannel (qns.entity.cchannel.cchannel.ClassicChannel): the inserting ClassicChannel

        """
        self.cchannels.append(cchannel)

    def get_cchannel(self, name: str) -> ClassicChannel:
        """Get the ClassicChannel by its name

        Args:
            name (str): its name
        Returns:
            the ClassicChannel

        """
        for n in self.cchannels:
            if n.name == name:
                return n
        raise IndexError(f"cchannel {name} does not exist")

    def add_memories(self, capacity: int = 0, decoherence_rate: float = 0, store_error_model_args: dict = {}):
        """Add quantum memories to every nodes in this network

        Args:
            capacity (int): the capacity of the quantum memory
            decoherence_rate (float): the decoherence rate
            store_error_model_args: the arguments for store_error_model

        """
        for idx, n in enumerate(self.nodes):
            m = QuantumMemory(name=f"m{idx}", node=n, capacity=capacity, decoherence_rate=decoherence_rate,
                              store_error_model_args=store_error_model_args)
            n.set_memory(m)

    def build_route(self):
        """Build static route tables for each nodes
        """
        self.route.build(self.nodes, self.qchannels)

    def query_route(self, src: QNode, dest: QNode) -> list[tuple[float, QNode, list[QNode]]]:
        """Query the metric, nexthop and the path

        Args:
            src: the source node
            dest: the destination node

        Returns:
            A list of route paths. The result should be sortted by the priority.
            The element is a tuple containing: metric, the next-hop and the whole path.

        """
        return self.route.query(src, dest)

    def add_request(self, src: QNode, dest: QNode, attr: dict = {}):
        """Add a request (SD-pair) to the network

        Args:
            src: the source node
            dest: the destination node
            attr: other attributions

        """
        raise NotImplementedError
        # req = Request(src=src, dest=dest, attr=attr)
        # self.requests.append(req)
        # src.add_request(req)
        # dest.add_request(req)

    def random_requests(self, number: int, allow_overlay: bool = False, attr: dict = {}):
        """Generate random requests

        Args:
            number (int): the number of requests
            allow_overlay (bool): allow a node to be the source or destination in multiple requests
            attr (Dict): request attributions

        """
        raise NotImplementedError
        # used_nodes: list[int] = []
        # nnodes = len(self.nodes)

        # if number < 1:
        #     raise QNSNetworkError("number of requests should be large than 1")

        # if not allow_overlay and number * 2 > nnodes:
        #     raise QNSNetworkError("Too many requests")

        # for n in self.nodes:
        #     n.clear_request()
        # self.requests.clear()

        # for _ in range(number):
        #     while True:
        #         src_idx = get_randint(0, nnodes - 1)
        #         dest_idx = get_randint(0, nnodes - 1)
        #         if src_idx == dest_idx:
        #             continue
        #         if not allow_overlay and src_idx in used_nodes:
        #             continue
        #         if not allow_overlay and dest_idx in used_nodes:
        #             continue
        #         if not allow_overlay:
        #             used_nodes.append(src_idx)
        #             used_nodes.append(dest_idx)
        #         break

        #     src = self.nodes[src_idx]
        #     dest = self.nodes[dest_idx]
        #     req = Request(src=src, dest=dest, attr=attr)
        #     self.requests.append(req)
        #     src.add_request(req)
        #     dest.add_request(req)


class QNSNetworkError(Exception):
    pass
