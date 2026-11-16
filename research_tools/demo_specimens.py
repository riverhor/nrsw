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

from .jpeg.statistics import metrics
from .r62_svk import runtime_sources


SPLIT = "f0_d3_o0"
METHOD = "NRSW_COMMON_UNIT"
TARGETS = (0.01, 0.05, 0.10)
COMPARISONS = {
    "nrsw": METHOD,
    "diagonal": "TAG_DIAGONAL",
    "genuine": "GENUINE_ENERGY",
    "concat": "CONCAT_NO_EVIDENCE",
}
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
EDIT_METHODS = {
    "fraud_crop_and_replace": ("Crop and replace", "裁切替换",
                               "The marked field was replaced with a crop from another document.",
                               "标出的字段使用另一张证件的裁切内容替换。"),
    "fraud_inpaint_and_rewrite": ("Inpaint and rewrite", "修补后重写",
                                  "The marked field was erased and rewritten with the same text.",
                                  "标出的字段抹除后重新绘制，文字内容相同。"),
}


@dataclass(frozen=True)
class Pair:
    field: str
    genuine: str
    forged: str
    formation: str
    same_jpeg_tables: bool


def example_pairs():
    source = Path(__file__).with_name("demo_specimen_pairs.json")
    selection = json.loads(source.read_bytes())
    if selection["context"] != SPLIT:
        raise ValueError("Specimen pairs must use the displayed held-out split")
    return [Pair(*item) for item in selection["pairs"]]


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

    model, _ = fit_model(training, data.offset, crossed=False)
    calibration_scores = model.score(calibration)
    cuts = np.sort(calibration_scores, axis=-1)[:, np.asarray(RANKS) - 1]
    with np.load(work / "all/refit/cal0.npz", allow_pickle=False) as archive:
        context = archive["contexts"].tolist().index(SPLIT)
        if archive["ids"].tolist() != list(calibration.ids):
            raise ValueError("Calibration document order differs from the paper run")
        columns = [model.methods.index(method) for method in NATIVE_METHODS]
        np.testing.assert_allclose(calibration_scores, archive["scores"][context, columns], rtol=0, atol=1e-7)
        np.testing.assert_allclose(cuts, archive["cuts"][context, columns], rtol=0, atol=1e-7)
    return data, training, model, test, cuts


def study_evidence(work):
    original_path = work / "all/refit/RESULTS.json"
    jpeg_root = work / "common-jpeg"
    jpeg_path = jpeg_root / "refit/RESULTS.json"
    verification_path = jpeg_root / "VERIFICATION.json"
    if not all(path.is_file() for path in (original_path, jpeg_path, verification_path)):
        return None

    original = json.loads(original_path.read_bytes())
    jpeg_data = jpeg_path.read_bytes()
    jpeg = json.loads(jpeg_data)
    verification = json.loads(verification_path.read_bytes())
    if original["scope"] != "all_384_contexts" or jpeg["status"] != "COMPLETE":
        raise ValueError("Expected the full original-image and common-JPEG studies")
    if verification["status"] != "PASS_R62_PARITY" or verification["result_sha256"] != sha256(jpeg_data).hexdigest():
        raise ValueError("Common-JPEG results must match the verified study")
    if jpeg["metric_order"][1:4] != ["ROC_FNR_1", "ROC_FNR_5", "ROC_FNR_10"] or jpeg["targets"] != list(TARGETS):
        raise ValueError("Unexpected common-JPEG ROC coordinates")
    first, second = (jpeg["methods"].index(COMPARISONS[key]) for key in ("nrsw", "diagonal"))
    jpeg_drop = np.subtract(jpeg["point"][second][1:4], jpeg["point"][first][1:4])
    native_drop = -np.array(original["results"]["comparisons"][COMPARISONS["diagonal"]]["roc_FNR"]["point"])
    return {
        "images": 16210,
        "native": (100 * native_drop).tolist(),
        "common_jpeg": (100 * jpeg_drop).tolist(),
    }


def held_out_evaluation(work, labels, groups, scores, cuts, methods):
    points, _ = metrics(labels, groups, scores, cuts)
    witness_path = Path(__file__).with_name("svk_native_witness.json")
    witness = json.loads(witness_path.read_bytes())
    if witness["context"] != SPLIT or witness["scope"]["test"] != len(labels):
        raise ValueError("Unexpected scope for the held-out result")
    comparisons = []
    for key, name in COMPARISONS.items():
        _, roc_fnr, fpr, fnr = np.split(points[methods.index(name)], [1, 4, 7])
        expected = witness["methods"][name]
        np.testing.assert_allclose([roc_fnr, fpr, fnr], [expected[k] for k in ("roc_FNR", "own_cut_FPR", "own_cut_FNR")],
                                   rtol=0, atol=2e-9)
        comparisons.append({
            "id": key,
            "rates": [{"target": target, "fpr": float(real_fpr), "fnr": float(real_fnr), "roc_fnr": float(roc)}
                      for target, real_fpr, real_fnr, roc in zip(TARGETS, fpr, fnr, roc_fnr, strict=True)],
        })
    result = {"genuine": int((labels == 0).sum()), "forged": int((labels == 1).sum()),
              "groups": len(set(groups)), "methods": comparisons}
    study = study_evidence(work)
    if study is not None:
        result["study"] = study
    return result


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
    data, training, model, test, all_cuts = prepared_model(work)
    method_index = model.methods.index(METHOD)
    cuts = all_cuts[method_index]
    fields = list(model.layout.fields)
    positions = {identity: index for index, identity in enumerate(test.ids)}
    pairs = example_pairs()
    wanted = [identity for pair in pairs for identity in (pair.genuine, pair.forged)]
    if len(set(wanted)) != len(wanted) or not set(wanted) <= positions.keys():
        raise ValueError("Example pairs must use distinct images from the reproduced test set")
    batch = test.subset(np.array([positions[identity] for identity in wanted]))
    if not batch.local_valid.all() or any(not data.documents[identity]["original_local_valid"] for identity in wanted):
        raise ValueError("Each example must have original field measurements")
    scores = model.score(batch)[method_index]
    full_teacher = model.routes[0][2].evidence
    evidence = full_teacher.evaluate(batch.local_features)
    weights = evidence["alpha"]
    if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Each document's field weights must add to one")
    with np.load(work / "all/refit/test0.npz", allow_pickle=False) as archive:
        context = archive["contexts"].tolist().index(SPLIT)
        if archive["ids"].tolist() != list(test.ids):
            raise ValueError("Test document order differs from the paper run")
        labels = np.array([data.documents[identity]["label"] for identity in test.ids], dtype=np.uint8)
        np.testing.assert_array_equal(labels, archive["labels"])
        np.testing.assert_array_equal(test.groups, archive["groups"])
        held_out_scores = archive["scores"][context, :len(model.methods)]
        archived_scores = held_out_scores[method_index, [positions[i] for i in wanted]]
        np.testing.assert_allclose(scores, archived_scores, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(scores[:, None] > cuts, archived_scores[:, None] > cuts)
    evaluation = held_out_evaluation(work, labels, test.groups, held_out_scores, all_cuts, model.methods)

    rows = {data.documents[identity]["id"]: data.documents[identity] for identity in wanted}
    if len({rows[pair.genuine]["group"] for pair in pairs}) != len(pairs):
        raise ValueError("Example pairs must use distinct source groups")
    plan = json.loads((work / "all/measurements/PLAN.json").read_text(encoding="utf-8"))
    plans = {row["id"]: row for row in plan["rows"] if row["id"] in rows}
    images = (work / "all/images").resolve(strict=True)
    examples, assets = [], {}
    for number, pair in enumerate(pairs, start=1):
        source_row, edited_row = rows[pair.genuine], rows[pair.forged]
        if source_row["label"] != 0 or edited_row["label"] != 1 or source_row["group"] != edited_row["group"]:
            raise ValueError("A pair must use the publisher's original and edited image from one source group")
        if not edited_row["member"].startswith(pair.formation + "/"):
            raise ValueError("Image does not match its publisher edit method")
        field = FIELD_LABELS[pair.field]
        field_zh = FIELD_LABELS_ZH[pair.field]
        method, method_zh, method_note, method_note_zh = EDIT_METHODS[pair.formation]
        image_rows = (plans[pair.genuine], plans[pair.forged])
        original, altered = (image_payload(images, row, fields) for row in image_rows)
        if (original[2] == altered[2]) != pair.same_jpeg_tables:
            raise ValueError("The pair's source JPEG quantization tables differ from the manifest")
        for label, row, (encoded, boxes, _) in zip(("genuine", "forged"), image_rows, (original, altered), strict=True):
            identifier = f"card-{number}-{label}"
            filename = f"{identifier}.jpg"
            assets[filename] = encoded
            index = wanted.index(row["id"])
            is_forged = label == "forged"
            examples.append({
                "id": identifier,
                "pair_id": f"card-{number}",
                "title": f"{field} edit ({method.lower()})" if is_forged else f"Unaltered card {number}",
                "title_zh": f"{field_zh}篡改（{method_zh}）" if is_forged else f"未篡改证件 {number}",
                "note": method_note if is_forged else "The original image from the dataset's synthetic identity pair.",
                "note_zh": method_note_zh if is_forged else "这张合成证件是数据集中的未篡改原图。",
                "label": label,
                "formation": pair.formation if is_forged else "positive",
                "same_jpeg_tables": pair.same_jpeg_tables,
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
        "selected_id": "card-1-genuine",
        "selected_fpr": TARGETS[0],
        "note": ("These held-out images use synthetic identities and portraits from IDSpace. NRSW scored the "
                 "original image files. Some pairs use different JPEG settings. Scores are not probabilities."),
        "note_zh": ("这些 IDSpace 留出测试图像使用合成身份及照片。NRSW 使用原始图像文件评分，"
                    "部分配对图像的 JPEG 编码设置不同。分数不等于概率。"),
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
        "evaluation": evaluation,
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
