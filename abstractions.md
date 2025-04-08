| **Aspect** | **Density Matrix (DM)** | **Bra-Ket (State Vector)** | **Stabilizer Formalism** | **Markov Process (Error Tracking)** | **Fidelity Tracking (Mixed States)** |
|------------|--------------------------|-----------------------------|---------------------------|--------------------------------------|---------------------------------------|
| **Core Idea** | Track full mixed state \(\rho\) | Track pure state \(|\psi⟩\) | Track Pauli stabilizers of state | Track Pauli errors as Markov transitions | Track fidelity to an ideal state (e.g., Bell state) |
| **Data Structure** | \(2^n \times 2^n\) complex matrix | \(2^n\) complex vector | Binary tableau of stabilizers | Pauli error label per qubit/photon | Scalar \(F\) per entangled pair |
| **Scalability** | ❌ Exponential (4ⁿ) | ⚠️ Exponential (2ⁿ) | ✅ Polynomial (n²) | ✅ Linear | ✅ Constant per link |
| **Supports Mixed States?** | ✅ Yes | ❌ No (pure states only) | ❌ No (pure stabilizer states) | ✅ Yes (via probabilistic errors) | ✅ Yes (by definition) |
| **Handles Noise?** | ✅ Fully (any noise model) | ❌ Not directly (needs switching to DM) | ⚠️ Only Pauli noise (within Clifford group) | ✅ Yes (Pauli error propagation) | ✅ Yes (simplified, assumed model) |
| **Handles Non-Clifford Gates?** | ✅ Yes | ✅ Yes | ❌ No (Clifford only) | ✅ Yes (if modeled as transition probabilities) | ⚠️ Only as fidelity degradation |
| **Handles Entanglement?** | ✅ Yes | ✅ Yes | ✅ Yes (stabilizer states) | ⚠️ Yes (implicitly via error history) | ✅ Yes (via fidelity to Bell state) |
| **Memory Usage** | 🚨 Very high | ⚠️ High | ✅ Efficient | ✅ Very efficient | ✅ Minimal |
| **Computation Speed** | 🐢 Slow | 🐇 Medium-fast | 🚗 Very fast | 🚗 Very fast | 🚀 Extremely fast |
| **Use Case Fit** | Precise sim, noise modeling, decoherence | Algorithm dev, ideal gate-level sim | Clifford circuits, QEC, optimizations | Quantum network + error tracking | High-level quantum networking, repeaters |
| **Examples of Tools** | Qiskit (Aer DM sim), QuTiP | Qiskit, Cirq, Braket, PennyLane | Stim, Qiskit (Clifford), LIQUi|> | QuISP, NetSquid | NetSquid, custom link-level sims |
| **Limitations** | Not scalable to large \(n\) | Can’t model mixed states or decoherence | Only for Clifford gates/states | Needs accurate error models | Assumes simplified state forms, approximate |