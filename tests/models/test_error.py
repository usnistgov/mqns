from collections.abc import Sequence

import numpy as np
import pytest

from mqns.models.core.state import (
    QUBIT_RHO_0,
    QUBIT_STATE_0,
    QUBIT_STATE_1,
    QUBIT_STATE_P,
    QubitState,
    check_qubit_rho,
    qubit_rho_equal,
    qubit_state_equal,
)
from mqns.models.epr import MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error import (
    BitFlipErrorModel,
    ChainErrorModel,
    CoherentErrorModel,
    DephaseErrorModel,
    DepolarErrorModel,
    DissipationErrorModel,
    ErrorModel,
    PerfectErrorModel,
    TimeDecayFunc,
    parse_time_decay,
)
from mqns.models.error.input import (
    ErrorModelDictLength,
    ErrorModelDictPError,
    ErrorModelDictPSurvival,
    ErrorModelDictTime,
    ErrorModelInputBasic,
    parse_error,
)
from mqns.models.qubit import Qubit
from mqns.models.qubit.gate import CNOT, H
from mqns.simulator import Time
from mqns.utils import rng


def test_parse_error():
    P_SURVIVAL, P_SURVIVAL2 = 0.9, 0.8
    P_ERROR = 1 - P_SURVIVAL
    T = 10.0
    RATE = -np.log(P_SURVIVAL) / T
    T2 = -np.log(P_SURVIVAL2) / RATE
    DICTS = [
        ErrorModelDictPSurvival(p_survival=P_SURVIVAL),
        ErrorModelDictPError(p_error=P_ERROR),
        ErrorModelDictTime(t=T, rate=RATE),
        ErrorModelDictLength(length=T, rate=RATE),
    ]

    # None and "PERFECT"
    for input in None, "PERFECT":
        error = parse_error(input, DepolarErrorModel, 0)
        assert isinstance(error, PerfectErrorModel)

    # ErrorModel instance
    error = parse_error(DepolarErrorModel().set(p_survival=P_SURVIVAL), DephaseErrorModel, 0)
    assert isinstance(error, DepolarErrorModel)
    assert error.p_survival == pytest.approx(P_SURVIVAL)

    # dict
    for d in DICTS:
        error = parse_error(d, DepolarErrorModel, 0)
        assert isinstance(error, DepolarErrorModel)
        assert error.p_survival == pytest.approx(P_SURVIVAL)

        if "rate" in d:
            error = parse_error({"rate": d["rate"]}, DepolarErrorModel, T2)
            assert error.p_survival == pytest.approx(P_SURVIVAL2)

    # ErrorModel instance and (possibly empty) dict
    for d in [{}] + DICTS:
        input = DephaseErrorModel().set(p_survival=P_SURVIVAL2)
        error = parse_error((input, d), DepolarErrorModel, 0)
        assert input.p_survival == pytest.approx(P_SURVIVAL2)

        assert isinstance(error, DephaseErrorModel)
        if len(d) > 0:
            assert error.p_survival == pytest.approx(P_SURVIVAL)
        else:
            assert input.p_survival == pytest.approx(P_SURVIVAL2)

    # ErrorModel type and dict
    for d in DICTS:
        error = parse_error((DephaseErrorModel, d), DepolarErrorModel, 0)
        assert isinstance(error, DephaseErrorModel)
        assert error.p_survival == pytest.approx(P_SURVIVAL)

        if "rate" in d:
            error.set(t=T2)
            assert error.p_survival == pytest.approx(P_SURVIVAL2)

    # str
    for prefix, typ in [
        ("BITFLIP", BitFlipErrorModel),
        ("DEPHASE", DephaseErrorModel),
        ("DEPOLAR", DepolarErrorModel),
        ("DISSIPATION", DissipationErrorModel),
    ]:
        # str with rate
        error = parse_error(f"{prefix}:{RATE}", PerfectErrorModel, 0)
        assert isinstance(error, typ)
        error.set(t=T)
        assert error.p_survival == pytest.approx(P_SURVIVAL)

        # str with p_error
        error = parse_error(f"{prefix}:{P_ERROR}", PerfectErrorModel, -1)
        assert isinstance(error, typ)
        assert error.p_survival == pytest.approx(P_SURVIVAL)

    with pytest.raises(ValueError, match="unrecognized ErrorModelInput string"):
        parse_error("UNKNOWN:0", PerfectErrorModel, 0)
    with pytest.raises(ValueError, match="unrecognized ErrorModelInput string.*:rate"):
        parse_error("DEPOLAR:x", PerfectErrorModel, 0)
    with pytest.raises(ValueError, match="unrecognized ErrorModelInput string.*:p_error"):
        parse_error("DEPOLAR:x", PerfectErrorModel, -1)

    # str concatenated
    error = parse_error(f"DEPOLAR:{P_ERROR}:DEPHASE:{1 - P_SURVIVAL2}", PerfectErrorModel, -1)
    assert isinstance(error, ChainErrorModel)
    assert isinstance(error.errors[0], DepolarErrorModel)
    assert error.errors[0].p_survival == pytest.approx(P_SURVIVAL)
    assert isinstance(error.errors[1], DephaseErrorModel)
    assert error.errors[1].p_survival == pytest.approx(P_SURVIVAL2)


def test_parse_time_decay():
    ACCURACY = 1000

    def check_same(
        decay: TimeDecayFunc,
        times: Sequence[int],
        *errors: ErrorModel,
    ):
        e0 = MixedStateEntanglement()
        for t in times:
            decay(e0, Time(t, accuracy=ACCURACY))

        e1 = MixedStateEntanglement()
        for m in errors:
            e1.apply_error(m)

        assert np.all(e0.probv == e1.probv)

    t10 = Time(10, accuracy=ACCURACY)
    # 0.010 sec => 100 Hz
    # 0.020 sec => 50 Hz
    # 0.040 sec => 25 Hz

    # "PERFECT"
    check_same(
        parse_time_decay("PERFECT", t10),
        [20],
        PerfectErrorModel(),
    )

    # None
    check_same(
        parse_time_decay(None, t10),
        [20, 10],
        DephaseErrorModel().set(t=0.020, rate=100),
        DephaseErrorModel().set(t=0.010, rate=100),
    )

    # dict
    check_same(
        parse_time_decay({"t_cohere": 0.040}, t10),
        [20],
        DephaseErrorModel().set(t=0.020, rate=25),
    )
    check_same(
        parse_time_decay({"rate": 25}, t10),
        [20],
        DephaseErrorModel().set(t=0.020, rate=25),
    )

    # type + dict
    check_same(
        parse_time_decay((DepolarErrorModel, {"rate": 25}), t10),
        [20],
        DepolarErrorModel().set(t=0.020, rate=25),
    )

    # str
    check_same(
        parse_time_decay("DEPHASE:25", t10),
        [20],
        DephaseErrorModel().set(t=0.020, rate=25),
    )
    check_same(
        parse_time_decay("DEPHASE:-0.040", t10),
        [20],
        DephaseErrorModel().set(t=0.020, rate=25),
    )
    check_same(
        parse_time_decay("DEPOLAR:50:PERFECT:DEPHASE:-0.040", t10),
        [20],
        DepolarErrorModel().set(t=0.020, rate=50),
        DephaseErrorModel().set(t=0.020, rate=25),
    )


@pytest.mark.parametrize(
    ("error", "success", "success_atol"),
    [
        (None, 1000, 0),
        ({"p_error": 0.06}, 960, 30),  # should have 96% success rate
    ],
)
def test_measure_error(error: ErrorModelInputBasic, success: int, success_atol: int):
    cnt = 0
    for _ in range(1000):
        q0 = Qubit(measure_error=error)
        if q0.measure() == 0:
            cnt += 1
    assert np.isclose(cnt, success, atol=success_atol)


@pytest.mark.parametrize(
    ("error", "rho_input"),
    [
        (None, [[0.5, 0.5], [0.5, 0.5]]),
        ({"p_error": 0.18}, [[0.5, 0.38], [0.38, 0.5]]),
    ],
)
def test_operate_error(error: ErrorModelInputBasic, rho_input: np.typing.ArrayLike):
    q0 = Qubit(operate_error=error)
    H(q0)
    assert qubit_rho_equal(q0.state.rho, check_qubit_rho(np.array(rho_input, dtype=np.complex128)))


@pytest.mark.parametrize(
    ("operate_error", "measure_error", "success", "success_atol"),
    [
        (None, None, 100, 0),
        ({"p_error": 0.18}, {"p_error": 0.06}, 82, 15),  # should have 82.78% success rate
    ],
)
def test_entanglement(
    operate_error: ErrorModelInputBasic, measure_error: ErrorModelInputBasic, success: int, success_atol: int
):
    cnt = 0
    for _ in range(100):
        q0 = Qubit(operate_error=operate_error, measure_error=measure_error)
        q1 = Qubit(operate_error=operate_error, measure_error=measure_error)
        H(q0)
        CNOT(q0, q1)
        if q0.measure() == q1.measure():
            cnt += 1
    assert np.isclose(cnt, success, atol=success_atol)


@pytest.mark.parametrize(
    ("random", "decohered"),
    [
        (1.0, False),
        (0.0, True),
    ],
)
def test_dissipation(monkeypatch: pytest.MonkeyPatch, random: float, decohered: bool):
    q0 = Qubit(name="q0")
    q1 = Qubit(name="q1")
    H(q0)
    CNOT(q0, q1)

    error = DissipationErrorModel().set(p_error=0.1)
    monkeypatch.setattr(rng, "random", lambda: random)
    q0.apply_error(error)

    if decohered:
        assert q0.state is not q1.state
        assert qubit_rho_equal(q0.state.rho, QUBIT_RHO_0)
        assert q1.state.rho.diagonal() == pytest.approx([0.5, 0.5])
    else:
        assert q0.state is q1.state


@pytest.mark.parametrize(
    ("random", "state"),
    [
        (0.0, QUBIT_STATE_0),
        (0.5, QUBIT_STATE_P),
        (1.0, QUBIT_STATE_1),
    ],
)
def test_coherent(monkeypatch: pytest.MonkeyPatch, random: float, state: QubitState):
    q = Qubit()

    error = CoherentErrorModel(length=100, standard_lkm=50)
    monkeypatch.setattr(rng, "uniform", lambda a, b: a + random * (b - a))
    q.apply_error(error)

    pure_state = q.state.state()
    assert pure_state is not None
    assert qubit_state_equal(pure_state, state)


def test_chain():
    chain = ChainErrorModel(
        [
            e0 := DepolarErrorModel(),
            e1 := DephaseErrorModel(),
        ]
    )
    chain.set(p_survival=0.9)
    assert pytest.raises(TypeError, lambda: chain.p_survival)
    assert pytest.raises(TypeError, lambda: chain.p_error)

    assert chain.errors == [e0, e1]
    assert e0.p_survival == pytest.approx(0.9, abs=1e-6)
    assert e1.p_survival == pytest.approx(0.9, abs=1e-6)

    we = WernerStateEntanglement()
    we.apply_error(chain)
    assert we.w == pytest.approx(0.81, abs=1e-6)

    me = MixedStateEntanglement()
    me.apply_error(chain)
    assert me.probv == pytest.approx([0.813333, 0.12, 0.033333, 0.033333], abs=1e-6)
