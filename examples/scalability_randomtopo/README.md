# scalability_randomtopo: Simulator Runtime Comparison

This experiment measures how simulation performance and outcomes scale as the network increases.
Results and analysis from this experiment are included in [supplemental material to the MQNS paper](https://doi.org/10.1109/MIC.2026.3651126/mm1).

## Scenario Explanation

This experiment generates random topologies with various network sizes, and then performs entanglement swapping simulations on those topologies.
It aims to report how MQNS performs when scaled to larger networks, and includes comparison with the SeQUeNCe quantum network simulator.

In each simulation run:

1. The simulator generates a random topology with MQNS `RandomTopology` class.

    * In default configurations, the network has an average node degree of 2.5.
    * For each network size, the number of entanglement requests is chosen to be proportional to the number of nodes, with 20% of nodes involved as either a source or a destination of the requests, plus intermediate nodes.
    * Topology and requests generation are deterministic, if the same random seed is used.

2. The workload is simulated in either MQNS or SeQUenCE.

    * **srt_mqns.py** contains the MQNS implementation.
    * **srt_sequence.py** contains the SeQUeNCe implementation, which uses the same topology and requests through a conversion function.
    * Both implementations use proactive forwarding with Statistical multiplexing and SWAP-ASAP swapping policy.

3. Each simulation run reports wall-clock execution time, along with other metrics for verification.

    * A wall-clock time limit can be set for each simulation run.
      If a simulation run is still ongoing after the defined time limit, it would be forcibly terminated.
      The full execution time can be extrapolated based on the simulation timeline progress upon termination.
    * See `RunResult` class (in `srt_detail/defs.py`) for a definition of the file format.

The **srt_plot.py** script loads the intermediate output files from individual runs, and generates the final output as CSV and plots.

## Example Usage

Every Python script in this experiment accepts a *parameters file*.
See `Params` class (in `srt_detail/defs.py`) for a definition of the file format.
Each parameters file includes:

* simulation duration and time limit
* topological parameters (e.g. how many pairs of qubits per quantum channel)
* list of network sizes

There are two examples of the parameters file:

* `params-full.toml` tests the full range of network sizes, which corresponds to the scalability experiment reported in the MQNS paper supplemental material.
* `params-short.toml` tests only smaller network sizes and with shorter duration, which allows a quick demo in about 10 minutes.

The **run-sequential.sh** script orchestrates the execution.
To start the example, cd to this directory, and then execute some of these sample commands:

```bash
# install compatible version of SeQUeNCe (optional)
pip install -r requirements.txt

# run the short demo sequentially
bash run-sequential.sh params-short.toml ./output-short
```

## Parallel Execution

The **run-docker.sh** script orchestrates Docker based parallel execution.
To use this script:

1. You must have a multi-core system with CPU isolation, so that some cores are exclusively reserved for the experiments, while other cores are unreserved for background services and general usage.
2. You must have Docker installed and authorized to use reserved cores.
3. Write the reserved cores in the *parameters file* `cpuset_cpus` list.
   There must be at least `runs` cores.
4. Run the script within virtual environment: `bash run-docker.sh params-full.toml ./output-full`
