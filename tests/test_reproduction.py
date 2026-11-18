from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

from nrsw.replay_svk import RANKS, _fold_points
from nrsw.reproduce import BUDGET_DIR, CROSSED_SVK, RAW_RUNTIME, _Evidence, _roc_fnr, reproduce
from nrsw.cli import build_parser


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


def test_reproduce_rejects_unknown_archive(tmp_path):
    archive = tmp_path / "submission.zip"
    archive.write_bytes(b"Different package")

    with pytest.raises(ValueError, match="The study ZIP must have SHA-256"):
        reproduce(archive)


@pytest.mark.parametrize("option", ["--extract-image-runtime", "--extract-r5"])
def test_public_reproduction_command_accepts_current_and_original_runtime_options(option):
    args = build_parser().parse_args(["reproduce", "study.zip", option, "runtime.zip"])
    assert str(args.extract_r5) == "runtime.zip"


@pytest.mark.parametrize("study_data", [False, True])
def test_evidence_checks_actual_member_hashes_after_resolving_names(study_data):
    names = {
        f"{BUDGET_DIR}/sealed_scores/nrsw_v2/TEST_SCORES.npz":
            f"{BUDGET_DIR}/saved_scores/nrsw_v2/TEST_SCORES.npz",
        CROSSED_SVK: "idspace_estimators/weight_strength/independent_outcome_audit.json",
        RAW_RUNTIME: "submission_evidence/reproduction/image_reproduction.zip",
        "submission_evidence/svk/RESULTS.json": "submission_evidence/svk/RESULTS.json",
    }
    payload = b"preserved evidence bytes"
    members = {new if study_data else old: {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
               for old, new in names.items()}
    manifest = "PACKAGE_MANIFEST.json" if study_data else "PACKAGE_MANIFEST_R62.json"
    packet = BytesIO()
    with ZipFile(packet, "w") as archive:
        archive.writestr(manifest, json.dumps({"members": members}))
        for name in members:
            archive.writestr(name, payload)
    with ZipFile(packet) as archive:
        evidence = _Evidence(archive, study_data=study_data)
        for name in names:
            assert evidence.read(name) == payload
        first = evidence._member(next(iter(names)))
        evidence.manifest[first]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="Checksum mismatch"):
            evidence.read(next(iter(names)))
