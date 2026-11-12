import itertools
from collections import defaultdict, deque
from typing import override

import pytest

from mqns.entity.cchannel import ClassicChannel, extract_cchannel_args
from mqns.entity.memory import MemoryDecohereEvent, MemoryQubit, QuantumMemory, QuantumMemoryInitKwargs, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel, QuantumChannelInitKwargs
from mqns.models.epr import Entanglement, WernerStateEntanglement
from mqns.network.ll import (
    LinkLayer,
    LinkLayerInitKwargs,
    LLActivateEvent,
    LLDeactivateEvent,
    QubitEntangledEvent,
    QubitReleasedEvent,
)
from mqns.network.network import QuantumNetwork, TimingMode, TimingModeAsync
from mqns.simulator import event_handler, func_to_event
from mqns.utils import rng


class NetworkLayer(Application[QNode]):
    check_epr_creation_prior: float | None = None
    """
    If set, ``handle_entangle`` verifies that every EPR was created this duration earlier.
    """

    def __init__(self):
        super().__init__()
        self.release_after = deque[float | None]()
        """If non-empty, ``QubitReleasedEvent`` would be emitted after specified duration for the next entanglement."""
        self.entangle: list[float] = []
        """Entanglement events, each entry is event time."""
        self.group_entangle = defaultdict[int | None, list[float]](list)
        """Entanglement times per group identifier."""
        self.decohere: list[float] = []
        """Decoherence events, each entry is event time."""

    @override
    def install(self, node) -> None:
        self._application_install(node, QNode)
        self.memory = self.node.memory
        self.epr_type = self.node.network.epr_type

    @event_handler
    def handle_entangle(self, event: QubitEntangledEvent) -> None:
        mq, epr = self.memory.read(event.mq.addr, has=self.epr_type)
        assert mq is event.mq
        if self.check_epr_creation_prior is not None:
            t_create = epr.decohere_time - self.memory.t_cohere
            assert t_create == event.t - self.check_epr_creation_prior
        self.entangle.append(event.t.sec)
        self.group_entangle[event.gid].append(event.t.sec)

        try:
            release_after = self.release_after.popleft()
        except IndexError:
            return
        if release_after is not None:
            self.simulator.sched(func_to_event(event.t + release_after, self._release, mq))

    def _release(self, mq: MemoryQubit) -> None:
        self.memory.read(mq.addr, remove=True)
        mq.state = QubitState.RELEASE
        self.simulator.sched(QubitReleasedEvent(self.node, mq, t=self.simulator.tc))

    @event_handler
    def handle_decohere(self, event: MemoryDecohereEvent) -> None:
        self.decohere.append(event.t.sec)
        event.qubit.state = QubitState.RELEASE
        self.simulator.sched(QubitReleasedEvent(self.node, event.qubit, is_decoh=True, t=event.t))


def make_linear(
    n=2,
    *,
    ll_args=LinkLayerInitKwargs(),
    qchannel_args: QuantumChannelInitKwargs,
    memory_args: QuantumMemoryInitKwargs,
    timing: TimingMode = TimingModeAsync(),
    epr_type: type[Entanglement] = WernerStateEntanglement,
) -> tuple[QuantumNetwork, tuple[QuantumChannel, ...], tuple[LinkLayer, ...], tuple[NetworkLayer, ...]]:
    """
    Build a linear topology.
    """
    rng.reseed("env")
    net = QuantumNetwork(timing=timing, epr_type=epr_type)

    nodes: list[QNode] = []
    for i in range(1, n + 1):
        node = QNode(
            f"n{i}",
            apps=[NetworkLayer(), LinkLayer(**ll_args)],
            memory=QuantumMemory("m1", **memory_args),
        )
        net.add_node(node)
        nodes.append(node)

    channels: list[QuantumChannel] = []
    for i, (a, b) in enumerate(itertools.pairwise(nodes), start=1):
        net.add_cchannel(ClassicChannel(f"c{i}", **extract_cchannel_args(qchannel_args)), a, b)
        qc = QuantumChannel(f"q{i}", **qchannel_args)
        net.add_qchannel(qc, a, b)
        channels.append(qc)

    net.build_route()
    return (
        net,
        tuple(channels),
        tuple(node.get_app(LinkLayer) for node in nodes),
        tuple(node.get_app(NetworkLayer) for node in nodes),
    )


def activate_group(
    t0: float | None,
    t1: float | None,
    src: Application[QNode],
    dst: Application[QNode],
    *,
    gid: int,
    n=1,
    roles: tuple[bool, bool] = (True, False),
):
    """
    Schedule ``LLActivateEvent`` and ``LLDeactivateEvent``.

    Args:
        t0: When to activate.
        t1: When to deactivate.
        src: Any application on the first node, typically the primary node.
        dst: Any application on the second node, typically the secondary node.
        gid: Group identifier.
        n: How many qubits to allocate.
        roles: Role (pri/2nd) of the src and dst nodes.
    """
    simulator = src.simulator
    ch = src.node.get_qchannel(dst.node)

    if t0 is not None:
        t = simulator.time(sec=t0)
        ch.assign_memory_qubits(capacity=n)
        for node, is_primary in zip((src.node, dst.node), roles, strict=True):
            simulator.sched(
                LLActivateEvent(
                    node,
                    ch,
                    t=t,
                    gid=gid,
                    q_target=n,
                    is_primary=is_primary,
                )
            )

    if t1 is not None:
        t = simulator.time(sec=t1)
        for node in src.node, dst.node:
            simulator.sched(LLDeactivateEvent(node, ch, t=t, gid=gid))


def check_link_arch_delays(
    t: float,
    ch: QuantumChannel,
    *,
    attempt_interval: float,
    d_notify_pri: float,
    d_notify_2nd: float,
) -> None:
    """
    Verify LinkArch computed delays match the expectation.
    """
    simulator = ch.simulator

    def assert_la_delays() -> None:
        la = ch.link_arch
        assert la.attempt_interval == simulator.time(sec=attempt_interval)
        assert la.d_notify_pri == simulator.time(sec=d_notify_pri)
        assert la.d_notify_2nd == simulator.time(sec=d_notify_2nd)

    event = func_to_event(simulator.time(sec=t), assert_la_delays)
    event.priority = 10000
    simulator.sched(event)


def force_attempts(monkeypatch: pytest.MonkeyPatch, ll: LinkLayer, /, **kwargs: list[int]):
    """
    Force entanglements to succeed at k-th attempt.

    Args:
        ll: LinkLayer instance.
        kwargs: Attempts numbers for upcoming entanglements, keyed by peer node name.
    """

    def new_calc_attempts(qchannel: QuantumChannel) -> int:
        nonlocal kwargs
        peer = qchannel.find_peer(ll.node)
        l = kwargs[peer.name]
        this_attempts, *l = l
        kwargs[peer.name] = l
        return this_attempts

    monkeypatch.setattr(ll, "_calc_attempt", new_calc_attempts)
