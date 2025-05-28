# Multiverse Quantum Network Simulator

![Lint](https://github.com/amar-ox/dynamic-qnetsim/actions/workflows/lint.yml/badge.svg)

## Overview

**Multiverse** is a quantum network simulator designed to streamline the comparative evaluation of entanglement routing under dynamic and heterogeneous network conditions. It addresses the need for a unified, flexible framework for rapidly prototyping and benchmarking a wide range of entanglement distribution strategies and quantum network architectures. The simulator supports systematic exploration of routing algorithms, swapping strategies, purification schedules, and resource management techniques across diverse network scenarios.

This project is part of an ongoing research effort to evaluate the quantum networking approaches presented in our recent survey:
🔗 [Entanglement Routing in Quantum Networks: A Comprehensive Survey](https://ieeexplore.ieee.org/document/10882978)

> ⚠️ This is an active research and development project. Functionality and APIs are evolving.

---

## Based on SimQN

This project reuses components from [SimQN v0.1.5](https://github.com/qnslab/SimQN), which is licensed under the GNU General Public License v3.0.

This is *not* a fork of the official SimQN repository, but rather a standalone project that incorporates a snapshot of SimQN's implementation—specifically the discrete-event simulation engine, noise modeling framework, and code structure. Substantial modifications have been made to support dynamic routing protocols and enhanced entanglement management capabilities.

This project is therefore licensed under the GPLv3. See the LICENSE file for details.

While we are developing dedicated documentation tailored to this simulator, users can refer to [SimQN’s documentation](https://qnlab-ustc.github.io/SimQN/) in the meantime to understand the foundational models and architecture.

---

## Installation

This is a development version to be installed from source.

First, clone the repository:

```bash
git checkout https://github.com/amar-ox/dynamic-qnetsim.git
cd dynamic-qnetsim
```

Create a virtual environment:

```bash
python -m venv mqns
source mqns/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**Option 1: Install from wheel (local build)**

```bash
python -m build
pip install dist/mqns-0.1.0-py3-none-any.whl
```

**Option 2: Install in editable mode**

```bash
pip install -e .
```

---

## Example: Three-Node Simulation

The example `examples/3_nodes_thruput.py` simulates a linear three-node quantum network (`S → R → D`) to evaluate how memory coherence time affects end-to-end entanglement throughput.

It demonstrates:

* Entanglement generation over lossy fiber links (approximating Barrett-Kok protocol)
* Swapping at an intermediate node
* Statistical analysis over multiple runs with variable memory coherence

Run the example with:

```bash
cd examples
python 3_nodes_thruput.py > output.log
```

The script outputs a plot of the entanglement rate versus memory coherence time, and log messages in the `.log` file.

More examples and configuration options will be added as the simulator evolves.

---

Feel free to open issues for bug reports or feature suggestions.