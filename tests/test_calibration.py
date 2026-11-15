from fractions import Fraction
import math

import numpy as np
import pytest

from nrsw.calibration import calibrate_scores, choose_np_rank, component_error_rates


@pytest.mark.parametrize("denominator,rank", [(100, 499), (20, 484), (10, 462)])
def test_paper_calibration_ranks(denominator, rank):
    result = choose_np_rank(500, 1, denominator)
    assert result["rank"] == rank
    assert result["bound"] <= 0.05 < result["previous_bound"]


def test_calibration_minimal_bound_with_exact_enumeration():
    n, eta, delta = 17, Fraction(2, 5), Fraction(1, 10)
    actual = choose_np_rank(n, eta.numerator, eta.denominator, delta.numerator, delta.denominator)
    feasible = []
    for rank in range(1, n + 1):
        tail = sum(Fraction(math.comb(n, j)) * (1 - eta) ** j * eta ** (n - j) for j in range(rank, n + 1))
        if tail <= delta:
            feasible.append(rank)
    assert actual["rank"] == feasible[0]


def test_ties_are_accepted_and_unavailable_is_explicit():
    result = calibrate_scores(np.ones(500), eta="1/10", delta="1/20")
    assert result.rank == 462
    np.testing.assert_array_equal(result.predict([0.0, 1.0, 1.00001]), [0, 0, 1])
    impossible = calibrate_scores([0.0, 1.0], eta=0.01)
    assert impossible.rank is None and impossible.threshold is None
    assert not impossible.available
    with pytest.raises(ValueError, match="calibration unavailable"):
        impossible.predict([8.0])


def test_component_error_weights():
    result = component_error_rates([2, 2, -1, -1, -1], [0, 0, 0, 1, 1], ["a", "a", "b", "c", "d"], 0)
    assert result["FPR"] == 0.5
    assert result["FNR"] == 1.0
    assert result["FP"] == 2 and result["TN"] == 1
