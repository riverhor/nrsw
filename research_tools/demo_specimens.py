"""Export the public image examples from the reproduced IDSpace experiment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from .r62_svk import runtime_sources


SPLIT = "f0_d3_o0"
METHOD = "NRSW_COMMON_UNIT"
TARGETS = (0.01, 0.05, 0.10)
FIELD_LABELS = {
    "birthday": "Date of birth",
    "card_num": "Card number",
    "country_code": "Country code",
    "expire_date": "Expiry date",
    "face": "Portrait",
    "gender": "Sex marker",
    "given_name": "Given name",
    "issue_authority": "Issuing authority",
    "issue_date": "Issue date",
    "personal_num": "Personal number",
    "surname": "Surname",
}
FIELD_LABELS_ZH = {
    "birthday": "出生日期",
    "card_num": "证件号码",
    "country_code": "国家代码",
    "expire_date": "有效期至",
    "face": "证件照片",
    "gender": "性别标记",
    "given_name": "名字",
    "issue_authority": "签发机关",
    "issue_date": "签发日期",
    "personal_num": "个人编号",
    "surname": "姓氏",
}


@dataclass(frozen=True)
class Pair:
    field: str
    genuine: str
    forged: str
    formation: str


# Select distinct edits from publisher annotations in the held-out split. Each
# pair shares its JPEG quantization tables. Choices do not use detector scores.
PAIRS = (
    Pair("given_name", "12717", "2648", "fraud_crop_and_replace"),
    Pair("card_num", "6523", "4834", "fraud_crop_and_replace"),
    Pair("birthday", "8152", "1852", "fraud_crop_and_replace"),
)


def prepared_model(work):
    runtime = work / "r5"
    runtime_sources(runtime)
    sys.path.insert(0, str(runtime / "package"))
    from .svk_estimators import NATIVE_METHODS, RANKS, fit_model
    from .svk_measurements import Measurements

    data = Measurements(runtime / "metadata", work / "all/measurements")
    training = data.training(SPLIT)
    calibration = data.batch(SPLIT, "calibration")
    test = data.batch(SPLIT, "test")
    if len(calibration) != 500 or len(set(calibration.groups)) != 500:
        raise ValueError("Expected 500 independent calibration groups")
    if any(data.documents[identity]["label"] != 0 for identity in calibration.ids):
        raise ValueError("Calibration uses genuine examples only")
    if set(calibration.groups).intersection(test.groups):
        raise ValueError("Calibration and test must use separate document groups")

    column = NATIVE_METHODS.index(METHOD)
    model, _ = fit_model(training, data.offset, crossed=False)
    calibration_scores = model.score(calibration)[column]
    thresholds = np.sort(calibration_scores)[np.asarray(RANKS) - 1]
    with np.load(work / "all/refit/cal0.npz", allow_pickle=False) as archive:
        context = archive["contexts"].tolist().index(SPLIT)
        if archive["ids"].tolist() != list(calibration.ids):
            raise ValueError("Calibration document order differs from the paper run")
        np.testing.assert_allclose(calibration_scores, archive["scores"][context, column], rtol=0, atol=1e-7)
        np.testing.assert_allclose(thresholds, archive["cuts"][context, column], rtol=0, atol=1e-7)
    return data, training, model, test, thresholds


def image_boxes(matrices, fields, size):
    corners = np.array([[0, 0, 1], [224, 0, 1], [224, 224, 1], [0, 224, 1]])
    boxes = {}
    for field, matrix in zip(fields, matrices, strict=True):
        vertices = corners @ np.asarray(matrix).T
        xy = vertices[:, :2] / vertices[:, 2:] / size
        low, high = np.clip(xy.min(axis=0), 0, 1), np.clip(xy.max(axis=0), 0, 1)
        if not np.all(high > low):
            raise ValueError(f"Field {field!r} has no visible rectangle")
        boxes[field] = np.concatenate((low, high)).tolist()
    return boxes


def image_payload(images_root, row, fields):
    source = (images_root / row["member"]).resolve(strict=True)
    if not source.is_relative_to(images_root):
        raise ValueError("Image must come from the published experiment image directory")
    encoded = source.read_bytes()
    if sha256(encoded).hexdigest() != row["sha256"]:
        raise ValueError("Source image differs from the image used in the experiment")
    with Image.open(BytesIO(encoded)) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Field geometry requires unrotated publisher images")
        if image.format != "JPEG":
            raise ValueError("Expected a JPEG source image")
        boxes = image_boxes(row["maps"]["original"], fields, image.size)
        quantization = image.quantization
    return encoded, boxes, quantization


def bundle(work, output):
    data, training, model, test, cuts = prepared_model(work)
    fields = list(model.layout.fields)
    positions = {identity: index for index, identity in enumerate(test.ids)}
    wanted = [identity for pair in PAIRS for identity in (pair.genuine, pair.forged)]
    if len(wanted) != 6 or len(set(wanted)) != len(wanted) or not set(wanted) <= positions.keys():
        raise ValueError("Expected three disjoint pairs from the reproduced test set")
    batch = test.subset(np.array([positions[identity] for identity in wanted]))
    if not batch.local_valid.all() or any(not data.documents[identity]["original_local_valid"] for identity in wanted):
        raise ValueError("Each example must have original field measurements")
    scores = model.score(batch)[model.methods.index(METHOD)]
    full_teacher = model.routes[0][2].evidence
    evidence = full_teacher.evaluate(batch.local_features)
    weights = evidence["alpha"]
    if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Each document's field weights must add to one")
    with np.load(work / "all/refit/test0.npz", allow_pickle=False) as archive:
        context = archive["contexts"].tolist().index(SPLIT)
        if archive["ids"].tolist() != list(test.ids):
            raise ValueError("Test document order differs from the paper run")
        archived_scores = archive["scores"][context, model.methods.index(METHOD), [positions[i] for i in wanted]]
        np.testing.assert_allclose(scores, archived_scores, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(scores[:, None] > cuts, archived_scores[:, None] > cuts)

    rows = {data.documents[identity]["id"]: data.documents[identity] for identity in wanted}
    plan = json.loads((work / "all/measurements/PLAN.json").read_text(encoding="utf-8"))
    plans = {row["id"]: row for row in plan["rows"] if row["id"] in rows}
    images = (work / "all/images").resolve(strict=True)
    examples, assets = [], {}
    for number, pair in enumerate(PAIRS, start=1):
        source_row, edited_row = rows[pair.genuine], rows[pair.forged]
        if source_row["label"] != 0 or edited_row["label"] != 1 or source_row["group"] != edited_row["group"]:
            raise ValueError("A pair must use the publisher's original and edited image from one source group")
        if not edited_row["member"].startswith(pair.formation + "/"):
            raise ValueError("Image does not match its publisher edit method")
        field = FIELD_LABELS[pair.field]
        field_zh = FIELD_LABELS_ZH[pair.field]
        image_rows = (plans[pair.genuine], plans[pair.forged])
        original, altered = (image_payload(images, row, fields) for row in image_rows)
        if original[2] != altered[2]:
            raise ValueError("Both images in this pair must share JPEG quantization settings")
        for label, row, (encoded, boxes, _) in zip(("genuine", "forged"), image_rows, (original, altered), strict=True):
            identifier = f"card-{number}-{label}"
            filename = f"{identifier}.jpg"
            assets[filename] = encoded
            index = wanted.index(row["id"])
            is_forged = label == "forged"
            examples.append({
                "id": identifier,
                "pair_id": f"card-{number}",
                "title": f"{field} edit" if is_forged else f"Unaltered card {number}",
                "title_zh": f"{field_zh}篡改" if is_forged else f"未篡改证件 {number}",
                "note": (f"The publisher marks {field.lower()} as edited."
                         if is_forged else "The original image from the publisher's synthetic identity pair."),
                "note_zh": (f"发布方将{field_zh}标注为篡改字段。"
                            if is_forged else "这是发布方同一合成身份对应的未篡改原图。"),
                "label": label,
                "image": f"./specimens/{filename}",
                "image_sha256": row["sha256"],
                "source_member": row["member"],
                "boxes": boxes,
                "edited_fields": [pair.field] if is_forged else [],
                "score": float(scores[index]),
                "weights": weights[index].tolist(),
                "energies": evidence["energy"][index].tolist(),
            })
    trace = {
        "dataset": "specimen",
        "layout": "idspace-svk",
        "layout_name": "Slovak identity-card specimen",
        "layout_name_zh": "斯洛伐克身份证样本",
        "selected_id": "card-2-forged",
        "selected_fpr": TARGETS[0],
        "note": ("These are original test images from the IDSpace dataset, which uses synthetic identities "
                 "and generated portraits. The displayed scores and field weights were computed from these exact "
                 "images using NRSW. Scores are not probabilities."),
        "note_zh": ("这些是 IDSpace 的原始测试图像，使用合成身份和生成的证件照片。"
                    "NRSW 以页面显示的同一图像计算伪造分数和字段权重。分数不等于概率。"),
        "source": {
            "name": "IDSpace by Cactus Lab",
            "url": "https://huggingface.co/datasets/cactuslab/IDSpace/tree/a3de23b950d55a39fbd36cfaee16bd260c4d457d",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "fields": fields,
        "field_labels": FIELD_LABELS,
        "field_labels_zh": FIELD_LABELS_ZH,
        "examples": examples,
        "thresholds": [{"fpr": fpr, "threshold": float(cut)} for fpr, cut in zip(TARGETS, cuts, strict=True)],
        "training": {
            "genuine": int((training.labels == 0).sum()),
            "forged": int((training.labels == 1).sum()),
            "calibration": 500,
        },
    }
    output.mkdir(parents=True)
    for filename, encoded in assets.items():
        (output / filename).write_bytes(encoded)
    (output / "examples.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path, help="Completed IDSpace image reproduction directory")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the demo images and scores")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        bundle(args.work, args.output)


if __name__ == "__main__":
    main()
