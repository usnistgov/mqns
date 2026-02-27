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

from mqns.network.fw import RoutingController, RoutingPath


class ProactiveRoutingController(RoutingController):
    """
    Centralized control plane for proactive routing.
    Works with ``ProactiveForwarder`` on quantum nodes.
    """

    def __init__(
        self,
        paths: RoutingPath | list[RoutingPath] | None = None,
    ):
        """
        Args:
            paths: routing path(s) to be automatically installed when the application is initiated.
        """
        super().__init__()
        self.paths = [] if not paths else paths if isinstance(paths, list) else [paths]

    @override
    def install(self, node):
        super().install(node)

        # install pre-requested paths on QNodes
        for rp in self.paths:
            self.install_path(rp)
