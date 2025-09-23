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

from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()


ext_modules = [
    Extension("mqns.simulator.ts", ["mqns/simulator/ts.pyx"]),
    Extension("mqns.simulator.pool", ["mqns/simulator/pool.pyx"]),
    Extension("mqns.simulator.simulator", ["mqns/simulator/simulator.py"]),
    Extension("mqns.models.qubit.const", ["mqns/models/qubit/const.py"]),
    Extension("mqns.models.qubit.gate", ["mqns/models/qubit/gate.py"]),
    Extension("mqns.models.qubit.qubit", ["mqns/models/qubit/qubit.py"]),
    Extension("mqns.models.qubit.decoherence", ["mqns/models/qubit/decoherence.py"]),
    Extension("mqns.models.qubit.factory", ["mqns/models/qubit/factory.py"]),
    Extension("mqns.models.qubit.utils", ["mqns/models/qubit/utils.py"]),
    Extension("mqns.models.epr.bell", ["mqns/models/epr/bell.py"]),
    Extension("mqns.models.epr.entanglement", ["mqns/models/epr/entanglement.py"]),
    Extension("mqns.models.epr.maxed", ["mqns/models/epr/mixed.py"]),
    Extension("mqns.models.epr.werner", ["mqns/models/epr/werner.py"]),
    Extension("mqns.entity.cchannel.cchannel", ["mqns/entity/cchannel/cchannel.py"]),
    Extension("mqns.entity.qchannel.qchannel", ["mqns/entity/qchannel/qchannel.py"]),
    Extension("mqns.entity.qchannel.losschannel", ["mqns/entity/qchannel/losschannel.py"]),
    Extension("mqns.entity.operator.operator", ["mqns/entity/operator/operator.py"]),
    Extension("mqns.entity.memory.memory", ["mqns/entity/memory/memory.py"]),
    Extension("mqns.network.route.dijkstra", ["mqns/network/route/dijkstra.py"]),
    Extension("mqns.network.topology.topo", ["mqns/network/topology/topo.py"]),
    Extension("mqns.network.topology.basictopo", ["mqns/network/topology/basictopo.py"]),
    Extension("mqns.network.topology.gridtopo", ["mqns/network/topology/gridtopo.py"]),
    Extension("mqns.network.topology.linetopo", ["mqns/network/topology/linetopo.py"]),
    Extension("mqns.network.topology.randomtopo", ["mqns/network/topology/randomtopo.py"]),
    Extension("mqns.network.topology.treetopo", ["mqns/network/topology/treetopo.py"]),
    Extension("mqns.network.topology.waxmantopo", ["mqns/network/topology/waxmantopo.py"]),
    Extension("mqns.network.protocol.bb84", ["mqns/network/protocol/bb84.py"]),
    Extension("mqns.network.protocol.classicforward", ["mqns/network/protocol/classicforward.py"]),
    Extension("mqns.network.protocol.node_process_delay", ["mqns/network/protocol/node_process_delay.py"]),
]


setup(
    name="mqns",
    author="amar",
    version="0.1.0",
    description="A simulator for comparative evaluation of quantum routing strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/amar-ox/dynamic-qnetsim",
    exclude_package_data={"docs": [".gitkeep"]},
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules),
    setup_requires=["numpy", "cython", "pandas", "twine", "wheel"],
    install_requires=["numpy", "pandas"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
