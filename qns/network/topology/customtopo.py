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

from qns.entity.node.app import Application
from qns.entity.qchannel.qchannel import QuantumChannel
from qns.entity.cchannel.cchannel import ClassicChannel
from qns.entity.node.qnode import QNode
from qns.entity.node.controller import Controller
from typing import Dict, List, Optional, Tuple
from qns.network.topology import Topology
from qns.entity.memory.memory import QuantumMemory

from enum import Enum
from typing import Dict, List, Optional, Tuple
import itertools
import copy

class CustomTopology(Topology):
    """
    TopologyCreator processed the topology dict.
    """
    def __init__(self, topo: Dict = {}):
        super().__init__(0)
        self.topo = topo

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        qnl: List[QNode] = []
        qcl: List[QuantumChannel] = []

        # Create quantum nodes
        for node in self.topo['qnodes']:
            qn = QNode(node['name'])
            for app in node['apps']:
                qn.add_apps(app)
            qnl.append(qn)

        # Create quantum channels and assign memories with proper capacity
        for ch in self.topo['qchannels']:
            link = QuantumChannel(name=f"q_{ch['node1']},{ch['node2']}", **ch['parameters'])
            qcl.append(link)
            
            for qn in qnl:
                if qn.name == ch['node1'] or qn.name == ch['node2']:
                    # Attach quantum channel to nodes
                    qn.add_qchannel(link)

                    # Assign a new memory with channel capacity for each end-node
                    memory_args = copy.deepcopy(next(node['memory'] for node in self.topo['qnodes'] if node['name'] == qn.name))
                    memory_args['capacity'] = ch['capacity']       # Set memory capacity to channel capacity
                    m = QuantumMemory(name=link.name, node=qn, **memory_args)
                    qn.add_memory(m)

        if "controller" in self.topo:
            self.controller = Controller(name=self.topo['controller']['name'], apps=self.topo['controller']['apps'])
        
        self.qnl = qnl
        return qnl, qcl
    
    def add_cchannels(self):
        ccl: List[ClassicChannel] = []
        for ch in self.topo['cchannels']:
            link = ClassicChannel(name=f"c_{ch['node1']},{ch['node2']}", **ch['parameters'])
            ccl.append(link)

            for node in self.qnl + [self.controller]:
                if node and (node.name == ch['node1'] or node.name == ch['node2']):
                    # Attach classic channel to nodes
                    node.add_cchannel(link)

        return ccl