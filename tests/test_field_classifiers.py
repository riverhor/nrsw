from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from nrsw.api import TRAIN_ROLES, _folds, feature_batch, fit
from nrsw.core import center_residuals
from nrsw.demo import example_data
from nrsw.field_classifiers import fit_field_classifiers, ink_descriptor, train_tail_reference
from nrsw.kernel import TrainBatch


def _fit_inputs(training):
    roles = tuple(TRAIN_ROLES[role] for role in training.roles)
    tags = tuple(tuple(np.flatnonzero(mask).tolist()) for mask in training.candidate_mask)
    training_batch = TrainBatch(feature_batch(training), roles, _folds(training), tags)
    rng = np.random.default_rng(24)
    pixels = rng.uniform(0.0, .04, size=(*training.candidate_mask.shape, 7))
    pixels[training.candidate_mask] += .2
    return training_batch, pixels, np.ones(pixels.shape[:2], dtype=bool)


def test_document_refit_keeps_coordinates_and_uses_genuine_training_reference():
    training, _, _ = example_data()
    batch, pixels, measured = _fit_inputs(training)
    original = fit(training)
    model = fit_field_classifiers(batch, original, pixels, measured)
    document = model.document

    assert model.training_identity != original.training_identity
    np.testing.assert_array_equal(document.delta, original.delta)
    np.testing.assert_array_equal(document.offset, original.offset)
    np.testing.assert_array_equal(document.local_mean, original.local_mean)
    assert document.strength_unit == original.strength_unit
    assert not np.array_equal(document.head.w, original.head.w)
    for current, old in zip(document.teachers, original.teachers, strict=True):
        assert current.fit_ids == old.fit_ids
        assert current.bank_ids == old.bank_ids
        assert current.evidence.stats["residual_mode"] == "field_only"
        assert old.evidence.stats.get("residual_mode") is None

    reference = batch.features.subset(np.flatnonzero(training.roles == "anchor"))
    scores = document.score(reference)
    expected = train_tail_reference(scores)
    assert model.document_reference == expected
    serving = document.teachers[-1].evidence
    energy = serving.evaluate(reference.local_features)["energy"]
    residual = reference.local_features.astype(np.float64) - serving.means
    projected = residual @ serving.factor
    np.testing.assert_allclose(energy, (projected**2).sum(2) / serving.studentizers, rtol=1e-12)

    edited = reference.local_features.copy()
    edited[:, 0] += .5
    np.testing.assert_array_equal(serving.evaluate(edited)["energy"][:, 1:], energy[:, 1:])
    assert not np.array_equal(center_residuals(edited, serving.means)[:, 1:],
                              center_residuals(reference.local_features, serving.means)[:, 1:])

    changed = replace(batch, folds=np.where(batch.folds < 0, -1, 1 - batch.folds))
    with pytest.raises(ValueError, match="same training rows, features, folds and tags"):
        fit_field_classifiers(changed, original, pixels, measured)


def test_field_maximum_uses_available_measurements_and_ignores_zero_pixel_crops():
    training, _, test = example_data()
    batch, pixels, measured = _fit_inputs(training)
    measured[training.candidate_mask[:, -1], -1] = False
    model = fit_field_classifiers(batch, fit(training), pixels, measured)
    assert model.pixel_heads[-1] is None
    assert all(branch is not None for branch in model.signed_heads)
    assert model.pixel_heads[0] is not None

    query = feature_batch(test)
    values = np.zeros((len(query), query.layout.m, 7), dtype=np.float64)
    available = np.ones(values.shape[:2], dtype=bool)
    values[4, 0] = pixels[training.candidate_mask[:, 0], 0][0]
    available[0, :] = False
    expected = model.document_reference.standardize(model.document.score(query))
    for j, branch in enumerate(model.signed_heads):
        signed = branch.reference.standardize(branch.head.score(query.local_features[query.local_valid, j]))
        expected[query.local_valid] = np.maximum(expected[query.local_valid], signed)
    branch = model.pixel_heads[0]
    expected[4] = max(expected[4], branch.reference.standardize(branch.head.score(values[4:5, 0]))[0])
    np.testing.assert_array_equal(model.score(query, values, available), expected)
    disabled = replace(model, signed_heads=(None,) * query.layout.m, pixel_heads=(None,) * query.layout.m)
    np.testing.assert_array_equal(disabled.score(query, values, available), model.document.score(query))
    unscaled = replace(model, document_reference=None)
    np.testing.assert_array_equal(unscaled.score(query, values, available), model.document.score(query))


def test_pixel_reference_uses_population_standard_deviation_floor():
    reference = np.array([0.] * 24 + [1.])
    assert train_tail_reference(reference) is None
    pixel = train_tail_reference(reference, use_std_floor=True)
    assert pixel.upper == 0.0
    assert pixel.spread == reference.std(ddof=0)


def test_field_fit_accepts_single_coordinate_signed_descriptors():
    training, _, _ = example_data()
    training = replace(training, local_features=training.local_features[:, :, :1],
                       global_mobile=training.global_mobile[:, :1])
    batch, pixels, measured = _fit_inputs(training)
    model = fit_field_classifiers(batch, fit(training), pixels, measured)

    assert all(branch.head.w.shape == (1,) for branch in model.signed_heads)


def test_ink_descriptor_measures_native_crops():
    pytest.importorskip("cv2")
    assert ink_descriptor(np.full((52, 47), 245, dtype=np.uint8)) is None
    np.testing.assert_array_equal(ink_descriptor(np.full((48, 48), 245, dtype=np.uint8)), np.zeros(7))
    gray = np.full((58, 62), 245, dtype=np.uint8)
    gray[:8] = 0
    gray[12:55, 8:57] = 137
    measured = ink_descriptor(gray)
    assert measured.dtype == np.float64
    assert measured.shape == (7,)
    assert (measured > 0).all()


def test_aligned_extraction_uses_native_pixel_boxes_and_same_signed_crops(tmp_path):
    pytest.importorskip("cv2")
    image_type = pytest.importorskip("PIL.Image")
    from nrsw.extraction import describe_rows
    from nrsw.vision import AlignedLayoutRegistrar

    gray = np.full((60, 104), 245, dtype=np.uint8)
    gray[5:12, 4:60] = 0
    gray[18:46, 12:52] = 137
    image_path = tmp_path / "sample.png"
    image_type.fromarray(gray).save(image_path)
    registrar = AlignedLayoutRegistrar({
        "layout_id": "card",
        "field_order": ["wide", "narrow"],
        "image_size": [104, 60],
        "quads": {"wide": [[4, 4], [60, 4], [60, 56], [4, 56]],
                  "narrow": [[64, 4], [104, 4], [104, 56], [64, 56]]},
    }, tmp_path)
    mobile = SimpleNamespace(global_descriptor=lambda image: np.zeros(512, dtype=np.float32),
                             encode=lambda crops: np.repeat(
                                 np.asarray(crops).mean(axis=(1, 2, 3), dtype=np.float32)[:, None], 512, axis=1))
    sd = SimpleNamespace(encode=lambda images: np.zeros((len(images), 512), dtype=np.float32))
    encoders = SimpleNamespace(batch_size=1, mobile=mobile, sd=sd)
    rows = [{"id": "sample", "group": "sample", "image": image_path, "label": -1,
             "role": "inference", "candidate_fields": [], "inner_fold": None}]
    original = describe_rows(rows, registrar, encoders, {"test": True})
    extension = describe_rows(rows, registrar, encoders, {"test": True}, field_classifiers=True)

    np.testing.assert_array_equal(original.local_features, extension.local_features)
    np.testing.assert_array_equal(extension.field_available, [[True, True]])
    np.testing.assert_array_equal(extension.pixel_available, [[True, False]])
    np.testing.assert_array_equal(extension.pixel_features[0, 0], ink_descriptor(gray[4:56, 4:60]))
    np.testing.assert_array_equal(extension.pixel_features[0, 1], np.zeros(7))
