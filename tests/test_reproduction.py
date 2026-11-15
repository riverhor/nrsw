import numpy as np
import pytest

from nrsw.replay_svk import RANKS, _fold_points
from nrsw.reproduce import _roc_fnr, reproduce


def test_crossverify_roc_uses_whole_score_ties_and_preserves_multiplicity():
    scores = np.ones(8)
    labels = np.array([0, 1, 1, 1] * 2)
    records = np.repeat([0, 1], 4)
    draws = np.array([[1, 1], [0, 2], [2, 0]])

    np.testing.assert_allclose(_roc_fnr(scores, labels, records, draws), [[99, 95, 90]] * 3)


def test_svk_component_budgets_and_strict_cut_decisions():
    labels = np.array([0, 0, 0, 1, 1, 1])
    membership = np.array([0, 0, 1, 0, 0, 1])
    scores = np.array([[[2, 0, 1, 2, 1, 0]]])
    cuts = np.array([[[2, 1, 0]]])

    points, class_counts = _fold_points(scores, cuts, labels, membership)

    assert RANKS == (499, 484, 462)
    np.testing.assert_equal(class_counts, [2, 2])
    np.testing.assert_allclose(points[0], [[0.99, 0.95, 0.90], [0, 0.25, 0.75], [1, 0.75, 0.5]])


def test_r62_replay_rejects_unknown_archive(tmp_path):
    archive = tmp_path / "NRSW_R62_submission_package.zip"
    archive.write_bytes(b"Different package")

    with pytest.raises(ValueError, match="Expected the R62 submission ZIP with SHA-256"):
        reproduce(archive)
