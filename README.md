# Multiverse Quantum Network Simulator

![Build](https://github.com/usnistgov/mqns/actions/workflows/build.yml/badge.svg) ![Lint](https://github.com/usnistgov/mqns/actions/workflows/lint.yml/badge.svg)

## Overview

**Multiverse** is a quantum network simulator designed to streamline the comparative evaluation of entanglement routing under dynamic and heterogeneous network conditions. It addresses the need for a unified, flexible framework for rapidly prototyping and benchmarking a wide range of entanglement distribution strategies and quantum network architectures. The simulator supports systematic exploration of routing algorithms, swapping strategies, purification schedules, and qubit/resource management across diverse network scenarios.

![MQNS logo](docs/mqns-logo.svg)

This software is developed at the [Smart Connected Systems Division](https://www.nist.gov/ctl/smart-connected-systems-division) of the [National Institute of Standards and Technology](https://www.nist.gov/).

This project is part of an ongoing research effort to evaluate the quantum networking approaches presented in our survey:
🔗 [Entanglement Routing in Quantum Networks: A Comprehensive Survey](https://ieeexplore.ieee.org/document/10882978).

## Current Features

### Routing Model

* Assumes **proactive centralized routing** defined in the survey taxonomy.
* Global path computation at simulation start:
  * **Dijkstra's algorithm** for single-path routing.
  * **Yen’s algorithm** for multipath routing.
* Paths can be installed/uninstalled at quantum routers; simulation focuses on the **forwarding phase**.

### Forwarding Phase Components

* **External and internal phases**:
  * Synchronous and asynchronous modes.
  * Elementary entanglement generation.
  * Swapping and purification.

* **Swapping strategies**:
  * Sequential and Balanced Tree.
  * Parallel (swap-asap).
  * Per-path [heuristic-based ad-hoc strategies](https://arxiv.org/abs/2504.14040).

* **Qubit lifecycle management**:
  * Tracks reservation, entanglement, release, and intermediate states of a qubit.

* **Qubit-path multiplexing** for:
  * Single/multiple source-destination requests.
  * Single/multiple paths.
  * Includes **buffer-space** and **statistical multiplexing** schemes ([ref](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8163/1/Multiplexing-schemes-for-quantum-repeater-networks/10.1117/12.893272.short)) and other dynamic entanglement allocation strategies.

* **Memory management**:
  * Policies for:
    * Allocating qubits to paths.
    * Choosing which qubits to swap when multiple are available.

* **Purification schemes**:
  * Limited support for:
    * Selection of link/segment to purify on each path.
    * Number of purification rounds.
    * Bennett 96 protocol.

### Entanglement Link Model

* Elementary link modeled using:
  * Werner states EPR generation.
  * Probability-based sampling.
  * Estimated duration based on entanglement link protocols.

* Link architectures:
  * Detection-in-Midpoint with single-rail encoding using 2-round Barrett-Kok protocol.
  * Detection-in-Midpoint with dual-rail polarization encoding.
  * Sender-Receiver with dual-rail polarization encoding.
  * Source-in-Midpoint with dual-rail polarization encoding.

## Roadmap

* WIP: Full support of purification scheme with **number of rounds or threshold fidelity**.
* Make **memory management and qubit selection policies** configurable.
* Add **log visualization timeline** for better debugging and analysis.
* Enable **runtime path computation and reconfiguration** based on request arrivals.
* Support for:
  * **Reactive centralized routing**.
  * **Distributed proactive routing**.
  * **Distributed reactive routing**.
* Refactor codebase for:
  * Better modularity and extensibility.
  * Comparative evaluation of entanglement routing strategy combinations.

> ⚠️ This is an active research and development project. Functionality and APIs are evolving.

---

## Based on SimQN

This project reuses components from [SimQN v0.1.5](https://github.com/QNLab-USTC/SimQN), which is licensed under the GNU General Public License v3.0.

This is *not* a fork of the official SimQN repository, but rather a standalone project that incorporates a snapshot of SimQN's implementation—specifically the discrete-event simulation engine, noise modeling framework, and code structure. Substantial modifications have been made to support dynamic routing protocols and enhanced entanglement management capabilities.

This project is therefore licensed under the GPLv3. See the LICENSE file for details.

---

## Installation

This is a development version to be installed from source.

First, clone the repository:

```bash
git checkout https://github.com/usnistgov/mqns.git
cd mqns
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Option 1: Install from wheel (local build)

```bash
python -m build
pip install dist/mqns-0.1.0-py3-none-any.whl
```

### Option 2: Install in editable mode

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
