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

from typing import override

from mqns.network.fw import Forwarder
from mqns.network.protocol.event import ManageActiveChannels


class ProactiveForwarder(Forwarder):
    """
    ProactiveForwarder is the forwarder of QNodes and receives routing instructions from the controller.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    @override
    def handle_path_change(self, *, path_id, uninstall, r_neighbor, **_):
        """
        Process LinkLayer changes after a path has been installed or uninstalled.

        If a path has been installed and it has a right neighbor:

        1. Notify LinkLayer to start elementary EPR generation toward the right neighbor.

        If a path has been uninstalled and it has a right neighbor:

        1. Notify LinkLayer to stop elementary EPR generation toward the right neighbor.
        """
        if r_neighbor:
            # instruct LinkLayer to start/stop generating EPRs on the qchannel toward the right neighbor
            self.simulator.add_event(
                ManageActiveChannels(
                    self.node,
                    *r_neighbor,
                    path_id=path_id if self.mux.qubit_has_path_id() else None,
                    start=not uninstall,
                    t=self.simulator.tc,
                    by=self,
                )
            )
