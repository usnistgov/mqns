import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping

import pytest

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.memory import QubitState
from mqns.entity.node import Application, Controller, Node, QNode
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk
from mqns.network.builder import NetworkBuilder
from mqns.network.fw import Forwarder
from mqns.network.fw.fw_swap import ForwarderSwapProc
from mqns.network.network import QuantumNetwork, TimingPhase, sync_phase_handler
from mqns.network.protocol.consumer import Consumer
from mqns.network.protocol.event import QubitEntangledEvent, QubitReleasedEvent
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Simulator, Time, event_handler, func_to_event
from mqns.utils import AutoIncrementIdentifier, log, rng


class QubitReleaseReset(Application[QNode]):
    """
    Reset released qubits to RAW state.
    """

    def __init__(self):
        super().__init__()
        self.history: list[tuple[int, Time]] = []
        """Qubit address and release time."""

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self) -> None:
        self.node.memory.clear()

    @event_handler
    def handle_qubit_released(self, event: QubitReleasedEvent):
        self.history.append((event.qubit.addr, event.t))
        log.debug(f"{self}: RELEASE {event.qubit}")
        event.qubit.state = QubitState.RAW

    @property
    def last_t(self) -> Time:
        """Retrieve the timestamp of last release event."""
        if not self.history:
            return Time.SENTINEL
        return self.history[-1][1]


LA_ALWAYS = LinkArchAlways(LinkArchDimBk())


def make_network(
    b: NetworkBuilder,
    *,
    swap_table_leak_tol=0,
    end_time=10.0,
) -> tuple[QuantumNetwork, Simulator]:
    """
    Finish building a network for unit testing.

    Args:
        b: NetworkBuilder with topology defined and applications installed.
        swap_table_leak_tol: ForwarderSwapProc memory leak tolerance.
        end_time: Simulation end time in seconds.
    """
    rng.reseed("env")
    ForwarderSwapProc.table_leak_tol = swap_table_leak_tol

    net = b.make_network()

    for node in net.nodes:
        if not node.get_apps(LinkLayer):
            node.add_apps(QubitReleaseReset())

    simulator = Simulator(0.0, end_time, install_to=(log, net))
    return net, simulator


def collect_cpacket_counts(monkeypatch: pytest.MonkeyPatch, *, cp=False, dp=False, cmd=False) -> Mapping[str, int]:
    """
    Gather classic packet counts between node pairs.

    Args:
        cp: Whether to include control plane traffic.
        dp: Whether to include data plane traffic.
        cmd: Whether to include command name.

    Returns: Mapping from node name(s) to number of packets, with these entries:

    * "src-dst": Packets sent from src to dst.
    * "src-*": Packets sent from src to any destination.
    * "*-dst": Packets sent to dst from any source.
    * "src-dst:CMD", "src-*:CMD", "*-dst:CMD": Packets with specific command name, if ``cmd`` is True.
    """
    orig_send_cpacket = Node.send_cpacket
    d = defaultdict[str, int](lambda: 0)

    def send_cpacket(self: Node, next_hop: Node, pkt: ClassicPacket):
        if self is pkt.src and (cp if (Controller in (type(pkt.src), type(pkt.dest))) else dp):
            prefixes = [f"{pkt.src.name}-{pkt.dest.name}", f"{pkt.src.name}-*", f"*-{pkt.dest.name}"]
            suffixes = [""]
            if cmd and isinstance(msg := pkt.get(), dict) and "cmd" in msg:
                suffixes.append(f":{msg['cmd']}")
            for prefix, suffix in itertools.product(prefixes, suffixes):
                d[prefix + suffix] += 1
        orig_send_cpacket(self, next_hop, pkt)

    monkeypatch.setattr(Node, "send_cpacket", send_cpacket)
    return d


def print_node_counters(net: QuantumNetwork):
    for node in net.nodes:
        fw = node.get_app(Forwarder)
        cons = node.get_app(Consumer)
        print(f"{node.name} {fw.cnt}")
        for req_id, pcnt in cons.cnt.items():
            print(f"    #{req_id}: {pcnt}")


def check_fw_counters(net: QuantumNetwork, **kwargs: Iterable[int | Iterable[int]]) -> None:
    __tracebackhide__ = True
    errors: list[str] = []
    cnts = [(node.name, node.get_app(Forwarder).cnt) for node in net.nodes]
    for key, expected_vec in kwargs.items():
        for expected, (node, cnt) in zip(expected_vec, cnts, strict=True):
            actual = getattr(cnt, key)
            if isinstance(actual, list):
                actual = sum(actual)
            if actual != expected and (not isinstance(expected, Iterable) or actual not in expected):
                errors.append(f"{node}.{key} = {actual} != {expected}")
    assert not errors, "\n".join(errors)


_provide_entanglements_autoid = AutoIncrementIdentifier("Tpe_")


def provide_entanglements(
    *etgs: tuple[float | Iterable[float], Forwarder, Forwarder] | tuple[Iterable[float], Iterable[Forwarder]],
    transform_t: Callable[[float], float] = lambda t: t,
    fidelity=0.99,
):
    """
    Provide elementary entanglement(s) to the forwarders.

    Args:
        etgs: Entanglement creation time, forwarder on left side, forwarder on right side;
              or, list of times, linear chain of forwarders.
        transform_t: Transform timestamp from input unit to seconds; default is identity function i.e. input unit is seconds.
        fidelity: Initial fidelity.
    """

    def make_entangle(src: Forwarder, dst: Forwarder):
        simulator = src.simulator
        t_creation = simulator.tc
        ch = src.node.get_qchannel(dst.node)
        la = ch.link_arch

        la.set(
            time_accuracy=simulator.accuracy,
            ch=ch,
            reset_time=0,
            tau_0=0,
            epr_type=src.network.epr_type,
        )

        ll_key = _provide_entanglements_autoid()
        epr = src.network.epr_type(
            decohere_time=t_creation + min(src.memory.t_cohere, dst.memory.t_cohere),
            fidelity_time=t_creation,
            src=src.node,
            dst=dst.node,
            mem_keys=(ll_key, ll_key),
            store_decays=(src.memory.time_decay, dst.memory.time_decay),
        )
        epr.fidelity = fidelity

        for node, neighbor, d_notify in (src, dst, la.d_notify_pri), (dst, src, la.d_notify_2nd):
            q, _ = next(node.memory.find(lambda _, v: v is None, qchannel=ch), (None, None))
            assert q is not None, f"insufficient qubits assigned to {ch}"
            q._state = QubitState.ENTANGLED0
            q.key = ll_key
            node.memory.write(q.addr, epr)
            simulator.sched(QubitEntangledEvent(node.node, neighbor.node, q, t=t_creation + d_notify))

    def sched_entangle(t: float, src: Forwarder, dst: Forwarder):
        if t < 0:
            return
        simulator = src.simulator
        simulator.sched(func_to_event(simulator.time(sec=transform_t(t)), make_entangle, src, dst))

    for etg in etgs:
        if len(etg) == 3:
            times, src, dst = etg
            if isinstance(times, int | float):
                sched_entangle(times, src, dst)
            else:
                for t in times:
                    sched_entangle(t, src, dst)
        else:
            times, fws = etg
            for t, (src, dst) in zip(times, itertools.pairwise(fws), strict=True):
                sched_entangle(t, src, dst)
