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
    PathActivateEvent,
    PathDeactivateEvent,
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
        self.path_entangle = defaultdict[int | None, list[float]](list)
        """Entanglement times per path_id."""
        self.decohere: list[float] = []
        """Decoherence events, each entry is event time."""

    @override
    def install(self, node) -> None:
        self._application_install(node, QNode)
        self.memory = self.node.memory
        self.epr_type = self.node.network.epr_type

    @event_handler
    def handle_entangle(self, event: QubitEntangledEvent) -> None:
        mq, epr = self.memory.read(event.qubit.addr, has=self.epr_type)
        assert mq is event.qubit
        if self.check_epr_creation_prior is not None:
            t_create = epr.decohere_time - self.memory.t_cohere
            assert t_create == event.t - self.check_epr_creation_prior
        self.entangle.append(event.t.sec)
        self.path_entangle[mq.path_id].append(event.t.sec)

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


def make_2nodes(
    *,
    ll_args=LinkLayerInitKwargs(),
    qchannel_args: QuantumChannelInitKwargs,
    memory_args: QuantumMemoryInitKwargs,
    timing: TimingMode = TimingModeAsync(),
    epr_type: type[Entanglement] = WernerStateEntanglement,
) -> tuple[QuantumNetwork, QuantumChannel, LinkLayer, LinkLayer, NetworkLayer, NetworkLayer]:
    rng.reseed("env")
    net = QuantumNetwork(timing=timing, epr_type=epr_type)
    net.add_node(n1 := QNode("n1", apps=[NetworkLayer(), LinkLayer(**ll_args)], memory=QuantumMemory("m1", **memory_args)))
    net.add_node(n2 := QNode("n2", apps=[NetworkLayer(), LinkLayer(**ll_args)], memory=QuantumMemory("m2", **memory_args)))
    net.add_cchannel(ClassicChannel("cc", **extract_cchannel_args(qchannel_args)), n1, n2)
    net.add_qchannel(qc := QuantumChannel("qc", **qchannel_args), n1, n2)
    net.build_route()

    return net, qc, n1.get_app(LinkLayer), n2.get_app(LinkLayer), n1.get_app(NetworkLayer), n2.get_app(NetworkLayer)


def activate_path(
    t0: float | None, t1: float | None, src: Application[QNode], dst: Application[QNode], *, path_id: int | None, n=1
):
    """
    Schedule ``PathActivateEvent`` and ``PathDeactivateEvent``.
    """
    simulator = src.simulator
    ch = src.node.get_qchannel(dst.node)
    ch.assign_memory_qubits(capacity=n, path_id=path_id)

    if t0 is not None:
        t = simulator.time(sec=t0)
        simulator.sched(PathActivateEvent(src.node, ch, path_id, t=t, is_primary=True))
        simulator.sched(PathActivateEvent(dst.node, ch, path_id, t=t, is_primary=False))

    if t1 is not None:
        t = simulator.time(sec=t1)
        simulator.sched(PathDeactivateEvent(src.node, ch, path_id, t=t))
        simulator.sched(PathDeactivateEvent(dst.node, ch, path_id, t=t))


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
        kwargs: Attempts numbers for upcoming entanglements, keyed by partner name.
    """

    def new_calc_attempts(qchannel: QuantumChannel) -> int:
        nonlocal kwargs
        partner = qchannel.find_peer(ll.node)
        l = kwargs[partner.name]
        this_attempts, *l = l
        kwargs[partner.name] = l
        return this_attempts

    monkeypatch.setattr(ll, "_calc_attempt", new_calc_attempts)
