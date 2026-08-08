"""
Automated validation for signs A–J using the fixed inference pipeline.

Tests each letter with a sample dataset image and reports accuracy metrics.
Also simulates the webcam orientation mismatch (vertical → rotate → predict).
"""

import os
import random
from typing import Optional

import cv2
import numpy as np

from backend.model_utils import SignLanguageClassifier
from backend.inference_pipeline import (
    PredictionStabilizer,
    CROP_PADDING_FRACTION,
    CONFIDENCE_THRESHOLD,
    prepare_roi_for_inference,
    apply_confidence_filter,
)


SIGNS = list("ABCDEFGHIJ")
DATASET_DIR = "dataset"


def _find_sample_image(class_dir: str) -> Optional[str]:
    """Pick one representative image for a class folder."""
    if not os.path.isdir(class_dir):
        return None
    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    if not images:
        return None
    random.seed(42)
    return os.path.join(class_dir, random.choice(images))


def validate_direct(classifier: SignLanguageClassifier) -> list[dict]:
    """Validate horizontal dataset images (no rotation — already in training orientation)."""
    rows = []
    for sign in SIGNS:
        class_dir = os.path.join(DATASET_DIR, sign.lower())
        sample = _find_sample_image(class_dir)
        if sample is None:
            rows.append({
                "actual": sign,
                "predicted": "MISSING",
                "confidence": 0.0,
                "match": False,
                "source": "N/A",
            })
            continue

        img = cv2.imread(sample)
        raw_letter, raw_conf, _, _ = classifier.predict_detailed(img)
        filtered = apply_confidence_filter(raw_letter, raw_conf)
        rows.append({
            "actual": sign,
            "predicted": filtered,
            "raw_predicted": raw_letter,
            "confidence": raw_conf,
            "match": filtered == sign,
            "source": os.path.basename(sample),
        })
    return rows


def validate_webcam_simulation(classifier: SignLanguageClassifier) -> list[dict]:
    """
    Simulate webcam pipeline: rotate dataset image to vertical (webcam orientation),
    then apply 90° CW rotation fix before classification.
    """
    rows = []
    stabilizer = PredictionStabilizer()
    for sign in SIGNS:
        class_dir = os.path.join(DATASET_DIR, sign.lower())
        sample = _find_sample_image(class_dir)
        if sample is None:
            rows.append({
                "actual": sign,
                "predicted": "MISSING",
                "confidence": 0.0,
                "match": False,
                "source": "N/A",
            })
            continue

        img = cv2.imread(sample)
        # Simulate vertical webcam capture (wrist bottom, fingers up)
        vertical_roi = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Warm stabilizer with repeated frames (simulates holding sign steady)
        stabilizer.reset()
        final = None
        for _ in range(15):
            final = stabilizer.predict_from_roi(classifier, vertical_roi, apply_rotation=True)

        rows.append({
            "actual": sign,
            "predicted": final["letter"],
            "raw_predicted": final["raw_letter"],
            "confidence": final["confidence"],
            "match": final["letter"] == sign,
            "source": os.path.basename(sample),
        })
    return rows


def _print_table(title: str, rows: list[dict]):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"{'Actual':<8} | {'Predicted':<10} | {'Raw':<8} | {'Confidence':<12} | {'Match'}")
    print("-" * 70)
    correct = 0
    for row in rows:
        match = "Yes" if row["match"] else "No"
        if row["match"]:
            correct += 1
        raw = row.get("raw_predicted", row["predicted"])
        print(
            f"{row['actual']:<8} | {row['predicted']:<10} | {raw:<8} | "
            f"{row['confidence']*100:6.2f}%     | {match}"
        )
    acc = correct / len(rows) * 100 if rows else 0
    print("-" * 70)
    print(f"Accuracy: {correct}/{len(rows)} ({acc:.1f}%)")
    return acc


def validate_rotation_fix(classifier: SignLanguageClassifier) -> dict:
    """Verify rotation fix restores accuracy on a known-good horizontal image."""
    sample = _find_sample_image(os.path.join(DATASET_DIR, "a"))
    if not sample:
        return {"status": "skipped"}

    img = cv2.imread(sample)
    vertical = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    _, conf_no_fix = classifier.predict(vertical)
    rotated = prepare_roi_for_inference(vertical, apply_rotation=True)
    pred_fixed, conf_fixed = classifier.predict(rotated)

    return {
        "sample": os.path.basename(sample),
        "vertical_no_fix_confidence": conf_no_fix,
        "vertical_with_fix_prediction": pred_fixed,
        "vertical_with_fix_confidence": conf_fixed,
        "rotation_fix_works": pred_fixed == "A" and conf_fixed >= CONFIDENCE_THRESHOLD,
    }


def validate_padding_constant() -> dict:
    return {
        "previous_padding": 0.18,
        "current_padding": CROP_PADDING_FRACTION,
        "expected_hand_area_pct": "25-35%",
    }


def main():
    print("SignBridge Inference Pipeline Validation")
    print(f"Config: padding={CROP_PADDING_FRACTION}, confidence_threshold={CONFIDENCE_THRESHOLD}")

    classifier = SignLanguageClassifier()
    if classifier.model is None:
        print("ERROR: Model not loaded. Cannot validate.")
        return

    # Task 7: Validate A–J
    direct_rows = validate_direct(classifier)
    direct_acc = _print_table("DIRECT DATASET VALIDATION (horizontal, no rotation)", direct_rows)

    webcam_rows = validate_webcam_simulation(classifier)
    webcam_acc = _print_table(
        "WEBCAM SIMULATION VALIDATION (vertical ROI -> 90 CW fix -> stabilize)", webcam_rows
    )

    # Task 8: Root cause verification
    print(f"\n{'=' * 70}")
    print("ROOT CAUSE VERIFICATION")
    print(f"{'=' * 70}")

    rot = validate_rotation_fix(classifier)
    pad = validate_padding_constant()

    print(f"\n1. Rotation fix:")
    print(f"   Sample: {rot.get('sample', 'N/A')}")
    print(f"   Vertical (no fix) confidence: {rot.get('vertical_no_fix_confidence', 0)*100:.1f}%")
    print(f"   Vertical + 90° CW fix: {rot.get('vertical_with_fix_prediction')} "
          f"({rot.get('vertical_with_fix_confidence', 0)*100:.1f}%)")
    print(f"   Fix effective: {rot.get('rotation_fix_works', False)}")

    print(f"\n2. Scale/padding fix:")
    print(f"   Before: {pad['previous_padding']} -> After: {pad['current_padding']}")
    print(f"   Target hand area: {pad['expected_hand_area_pct']}")

    print(f"\n3. Prediction stability:")
    print(f"   Majority vote window: 15 frames (PredictionStabilizer)")

    print(f"\n4. Confidence filtering:")
    print(f"   Threshold: {CONFIDENCE_THRESHOLD} → displays 'Unknown' below threshold")

    print(f"\n5. Summary:")
    print(f"   Direct dataset accuracy:  {direct_acc:.1f}%")
    print(f"   Webcam simulation accuracy: {webcam_acc:.1f}%")

    if webcam_acc >= 80:
        print("\n[OK] Application is READY for internship demonstration.")
    elif direct_acc >= 90:
        print("\n[WARN] Dataset accuracy is good; webcam simulation needs live testing with camera.")
    else:
        print("\n[FAIL] Further tuning may be needed before demonstration.")


if __name__ == "__main__":
    main()
