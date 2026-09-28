from typing import Any, Literal, cast

import pytest

from mqns.models.core import BASIS_X, BASIS_Z, Basis
from mqns.models.epr import EntangledQubitPair, Entanglement, MixedStateEntanglement, PurifProtocol, WernerStateEntanglement
from mqns.models.error import DepolarErrorModel, parse_time_decay
from mqns.models.qubit import CNOT, H, QState, Qubit
from mqns.simulator import Time
from mqns.utils import rng


def _make_qubit_pair_from_mixed_state(**kwargs):
    return EntangledQubitPair.move_from(MixedStateEntanglement(**kwargs))


@pytest.mark.parametrize("epr_type", [EntangledQubitPair, WernerStateEntanglement, MixedStateEntanglement])
def test_fidelity_error(epr_type: type[Entanglement]):
    """
    Verify that ``DepolarErrorModel`` has the same fidelity reduction across entanglement models.
    """
    epr = epr_type()
    assert epr.fidelity == pytest.approx(1.0, abs=1e-6)

    depolar = DepolarErrorModel().set(p_error=0.1)
    epr.apply_error(depolar)
    assert epr.fidelity == pytest.approx(0.925, abs=1e-6)


def test_fidelity_trace():
    """
    Verify that ``EntangledQubitPair`` could trace-out other qubits on a copy of density matrix
    while calculating fidelity.
    """
    # Construct a |Φ+> state with qubits in reversed order in the QState.
    epr0q1 = Qubit()
    epr0q0 = Qubit()
    H(epr0q1)
    CNOT(epr0q1, epr0q0)
    assert epr0q1.state.qubits == [epr0q1, epr0q0]
    epr0 = EntangledQubitPair(epr0q0, epr0q1)
    epr0.apply_error(DepolarErrorModel().set(p_error=0.1))
    # epr0.fidelity runs the transpose step.
    assert epr0.fidelity == pytest.approx(0.925, abs=1e-6)

    epr1 = EntangledQubitPair()
    QState.joint(epr0.q0, epr1.q0)
    epr1.apply_error(DepolarErrorModel().set(p_error=0.2))
    assert epr0.q0.state.num == 4

    # epr0.fidelity runs trace-out and transpose steps.
    assert epr0.fidelity == pytest.approx(0.925, abs=1e-6)
    # epr1.fidelity runs the trace-out step.
    assert epr1.fidelity == pytest.approx(0.850, abs=1e-6)
    assert epr0.q0.state.num == 4


@pytest.mark.parametrize("epr_type", [EntangledQubitPair, WernerStateEntanglement, MixedStateEntanglement])
@pytest.mark.parametrize("decohered", [False, True])
def test_fidelity_move(epr_type: type[Entanglement], decohered: bool):
    """
    Verify that when an entanglement model is transported into individual qubits and re-constructed
    into ``EntangledQubitPair``, the fidelity is preserved.
    """
    epr0 = epr_type()
    if decohered:
        epr0.is_decohered = True
    else:
        epr0.apply_error(DepolarErrorModel().set(p_error=0.1))
        assert epr0.fidelity == pytest.approx(0.925, abs=1e-6)

    epr1 = EntangledQubitPair.move_from(epr0)
    assert epr0.is_decohered is True
    assert epr1.is_decohered is False  # is_decohered is not copied
    if decohered:
        assert epr1.fidelity == pytest.approx(0.5, abs=1e-6)
    else:
        assert epr1.fidelity == pytest.approx(0.925, abs=1e-6)


@pytest.mark.parametrize("reversed", [False, True])
@pytest.mark.parametrize("epr_type", ["M", "Q"])
def test_swap(epr_type: Literal["M", "Q"], reversed: bool):
    """
    Verify that swapping ``EntangledQubitPair`` and ``MixedStateEntanglement`` yield the same fidelity.
    Note that we cannot compare with ``WernerStateEntanglement`` because the memory ``DephaseErrorModel``
    would become depolarization on Werner state, which reduces fidelity differently.
    """
    t0 = Time.from_sec(0.0, accuracy=1000000)
    t_cohere = Time.from_sec(1.0, accuracy=1000000)  # memory dephasing time: 1 second
    decay = parse_time_decay(None, t_cohere)

    if epr_type == "M":
        make_epr = MixedStateEntanglement
    else:
        make_epr = _make_qubit_pair_from_mixed_state

    epr0 = make_epr(fidelity=0.94, fidelity_time=t0, decohere_time=t0 + t_cohere, store_decays=(decay, decay))
    epr1 = make_epr(fidelity=0.91, fidelity_time=t0, decohere_time=t0 + t_cohere, store_decays=(decay, decay))
    if reversed:
        epr0, epr1 = epr1, epr0

    ne, local_success = Entanglement.swap(epr0, epr1, now=Time.from_sec(0.1, accuracy=1000000))
    assert local_success is True
    assert ne.fidelity == pytest.approx(0.723745, abs=1e-6)


@pytest.mark.parametrize("rng_success", [False, True])
@pytest.mark.parametrize(
    ("epr_type", "store_decay"),
    [
        ("W", "PERFECT"),
        ("M", "PERFECT"),
        ("M", None),
        ("Q", "PERFECT"),
        ("Q", None),
    ],
)
@pytest.mark.parametrize(
    ("proto", "basis", "perfect_fidelity", "dephase_fidelity"),
    [
        pytest.param(PurifProtocol.BBPSSW, BASIS_Z, 0.946065, 0.744208, id="BBPSSW-Z"),
        pytest.param(PurifProtocol.BBPSSW, BASIS_X, 0.946065, 0.907167, id="BBPSSW-X"),
        pytest.param(PurifProtocol.DEJMPS, BASIS_Z, 0.946065, 0.907167, id="DEJMPS-Z"),
        pytest.param(PurifProtocol.DEJMPS, BASIS_X, 0.946065, 0.907167, id="DEJMPS-X"),
    ],
)
def test_purify(
    monkeypatch: pytest.MonkeyPatch,
    rng_success: bool,
    epr_type: Literal["W", "M", "Q"],
    proto: PurifProtocol,
    basis: Basis,
    store_decay: Literal["PERFECT", None],
    perfect_fidelity: float,
    dephase_fidelity: float,
):
    """
    Verify that purification on each entanglement model yields the same fidelity.
    """

    def new_random() -> float:
        nonlocal randoms
        rnd, *randoms = randoms
        return rnd

    randoms: list[float] = []
    monkeypatch.setattr(rng, "random", new_random)

    if epr_type == "W":
        make_epr = WernerStateEntanglement
        randoms += [0] if rng_success else [1]
    elif epr_type == "M":
        make_epr = MixedStateEntanglement
        randoms += [0] if rng_success else [1]
    else:
        make_epr = _make_qubit_pair_from_mixed_state
        randoms += [1, 1] if rng_success else [1, 0]

    t0 = Time.from_sec(0.000, accuracy=1000000)
    t1 = Time.from_sec(0.010, accuracy=1000000)
    t_cohere = Time.from_sec(0.100, accuracy=1000000)
    decay = parse_time_decay(store_decay, t_cohere)
    epr0 = make_epr(fidelity=0.94, fidelity_time=t0, decohere_time=t0 + t_cohere, store_decays=(decay, decay))
    epr1 = make_epr(fidelity=0.91, fidelity_time=t1, decohere_time=t1 + t_cohere, store_decays=(decay, decay))

    success = epr0.purify(cast(Any, epr1), now=Time.from_sec(0.020, accuracy=1000000), protocol=proto, basis=basis)
    assert epr1.is_decohered
    assert success is rng_success
    if success:
        assert epr0.fidelity == pytest.approx(perfect_fidelity if store_decay == "PERFECT" else dephase_fidelity)
