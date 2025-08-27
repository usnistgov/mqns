Introduction
============

Quantum networks (QNs) aim to connect quantum processors across long distances to enable
applications such as **distributed quantum computing, secure communication, and enhanced sensing**.
All of these depend on the **generation and distribution of entanglement** between distant nodes.
However, this process is fragile: photon loss, decoherence, limited quantum memories, and
hardware heterogeneity create formidable challenges.

To achieve reliable end-to-end entanglement distribution, QNs require strategies that combine:

- **Routing** — proactive or reactive approaches for selecting entanglement paths
- **Swapping** — orchestrating intermediate nodes to extend entanglement across hops
- **Purification** — improving fidelity of noisy entangled states
- **Memory management** — deciding how qubits are allocated, reserved, and multiplexed

These operations are tightly coupled, and their interactions must be studied holistically.
Unfortunately, many existing simulators either assume a specific network architecture,
focus narrowly on quantum state evolution, or lack support for runtime reconfiguration.

---

Motivation for Multiverse
-------------------------

**Multiverse** was created to fill this gap by offering a **discrete-event simulation environment**
dedicated to *network-layer evaluations* of entanglement distribution.  

Its design is guided by three principles:

1. **Flexibility** – support proactive, reactive, or hybrid routing strategies without assuming
   a fixed stack.
2. **Extensibility** – modular components for time scheduling, memory management,
   and qubit lifecycles that can be reconfigured at runtime.
3. **Realism** – models grounded in physical entanglement link architectures
   (Detection-in-Midpoint, Sender-Receiver, Source-in-Midpoint, with or without transduction).

By focusing on modular abstractions rather than protocol-specific assumptions,
Multiverse enables **fair and reproducible evaluations** across diverse scenarios.
It also serves as a prototyping tool for discovering which network functions and interfaces
are most critical for future **quantum repeater networks**.

---

Background & Related Work
--------------------------

The field of QN simulation has evolved significantly:

- **SimulaQron (2018):** provided basic application-layer testing but lacked routing.
- **NetSquid (2021):** detailed state tracking and protocol interfaces, but computationally heavy.
- **SeQUeNCe (2021):** precise and modular, though limited in swapping/routing flexibility.
- **QuNetSim (2021):** focused on small-scale QNs and application testing.
- **QuISP (2022):** scalable, but tied to its dedicated RuleSet architecture.
- **SimQN (2023):** fidelity-based modeling with network topologies, but rigid in function design.

**Multiverse builds upon these efforts** by combining the extensibility of modular simulators with
an explicit focus on entanglement routing, purification, swapping, and memory policies—all under
heterogeneous and dynamic conditions.
