# extctrl: External Control Plane example

This example demonstrates how to create a split-horizon simulation, in which:

* The data plane is running in `extctrl_dp.py` Python script, implemented with MQNS library.
* The control plane is running as an external program.
* Control messages are delivered through NATS Messaging service.

## Scenario Explanation

The `extctrl_dp.py` Python script uses `NetworkBuilder` class to define a six-node topology with one shared link between S1-R1-R2-S2 and S2-R1-R2-S2 paths:

![topology](topo.svg)

The topology can be constructed in either proactive-centralized or reactive-centralized mode.
In either mode, a `Controller` node is defined, where a centralized controller application would be installed.

The `extctrl_dp.py` then calls `NetworkBuilder.external_controller()` method to convert the network into external control.
This method removes the centralized controller application (`ProactiveRoutingController` or `ReactiveRoutingController`), and replaces it with a `ClassicBridge` application.
The `ClassicBridge` application would then:

* Intercept classic packets sent to the controller and relay them to NATS JetStream for processing in the external program.
* Receive messages from NATS JetStream sent by the external program for injecting into the simulated network toward quantum nodes.

In this example, the external program is written in Rust as a Cargo binary crate.
It contains the simplified logic of a centralized controller application comparable to `ProactiveRoutingController` or `ReactiveRoutingController`.
This program connects to the NATS Messaging service, and can then communicate with quantum nodes using the JSON-based message formats defined by these applications.

The simulation clock is maintained in Python, but is synchronized with the external program through Conservative PDES (Parallel Discrete Event Simulation) technique.
See `ClassicBridge` and `ClassicConnector` class documentation for details.

## Example Usage

Apart from MQNS library, the following dependencies are required:

* Rust toolchain: see [Rust book](https://doc.rust-lang.org/stable/book/ch01-01-installation.html)
* NATS library in the Python venv: `pip install -r examples/extctrl/requirements.txt`
* NATS Messaging service with JetStream enabled: `docker run -d --name nats_server -p 4222:4222 nats:2 -addr :: -js`
* `nats` CLI client: see [release page](https://github.com/nats-io/natscli/releases)

The **run.sh** script orchestrates the execution.
It creates necessary NATS JetStream objects, and launches both Python script and Rust program with appropriate flags.
To start the example, cd to this directory, and then execute some of these sample commands:

```bash
# view help
bash run.sh --help

# default: proactive-centralized-async mode, enable S1-D1 path with l2r swap order
bash run.sh

# change simulation accuracy
bash run.sh --sim_accuracy 200000

# proactive-centralized-async mode, enable S1-D1 path with asap swap order
bash run.sh -- --mode PCA -- --path1 asap --path2 disabled

# proactive-centralized-async mode, enable both paths with r2l swap order, install paths at different times
bash run.sh -- --mode PCA -- --sim_duration 60 --path1 r2l --path1_i 0 --path2 r2l --path2_i 30
```

During the scenario execution, you can view NATS messages on a separate console:

```bash
nats sub 'mqns.classicbridge.>'
```
