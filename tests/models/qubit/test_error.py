import numpy as np
import pytest

from mqns.models.core.state import check_qubit_rho, qubit_rho_equal
from mqns.models.error import DissipationErrorModel, ErrorModelInput
from mqns.models.qubit import Qubit
from mqns.models.qubit.gate import CNOT, H
from mqns.utils import rng


@pytest.mark.parametrize(
    ("error", "success", "success_atol"),
    [
        (None, 1000, 0),
        ({"p_error": 0.06}, 960, 30),  # should have 96% success rate
    ],
)
def test_measure_error(error: ErrorModelInput, success: int, success_atol: int):
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
def test_operate_error(error: ErrorModelInput, rho_input: np.typing.ArrayLike):
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
def test_entanglement(operate_error: ErrorModelInput, measure_error: ErrorModelInput, success: int, success_atol: int):
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
    else:
        assert q0.state is q1.state
