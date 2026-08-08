import os
import random
import cv2
import numpy as np
from backend.model_utils import SignLanguageClassifier

def run_sanity_test():
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    if classifier.model is None:
        print("Error: Model could not be loaded.")
        return

    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return

    # Find all image files in the dataset
    all_images = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(root, file)
                # Actual label is the directory name
                dir_name = os.path.basename(root)
                actual_label = 'SPACE' if dir_name == '{' else dir_name.upper()
                all_images.append((full_path, actual_label))

    if not all_images:
        print("Error: No images found in dataset.")
        return

    print(f"Found {len(all_images)} total images in dataset.")

    # Select 50 random images
    random.seed(42)  # for reproducibility
    sampled_images = random.sample(all_images, 50)

    print("\n--- RUNNING SANITY TEST ON 50 RANDOM DATASET IMAGES ---")
    correct_count = 0
    total_conf = 0.0
    results_table = []
    misclassified = []

    for path, actual in sampled_images:
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read: {path}")
            continue

        pred, conf = classifier.predict(img)
        is_correct = (pred == actual)
        if is_correct:
            correct_count += 1
        else:
            misclassified.append({
                'path': path,
                'actual': actual,
                'predicted': pred,
                'confidence': conf
            })

        total_conf += conf
        results_table.append((os.path.basename(path), actual, pred, conf))

    # Print results table
    print("\n| Filename | Actual Label | Predicted Label | Confidence | Match |")
    print("|---|---|---|---|---|")
    for fname, actual, pred, conf in results_table:
        match = "Yes" if actual == pred else "No"
        print(f"| {fname} | {actual} | {pred} | {conf:.4f} ({conf*100:.2f}%) | {match} |")

    top1_acc = correct_count / 50.0
    avg_conf = total_conf / 50.0

    print(f"\n--- SUMMARY METRICS ---")
    print(f"Top-1 Accuracy:     {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"Average Confidence: {avg_conf:.4f} ({avg_conf*100:.2f}%)")

    if misclassified:
        print(f"\n--- MISCLASSIFIED IMAGES ({len(misclassified)}) ---")
        for mc in misclassified:
            print(f"File: {mc['path']}")
            print(f"  Actual:    {mc['actual']}")
            print(f"  Predicted: {mc['predicted']}")
            print(f"  Confidence:{mc['confidence']:.4f}")
    else:
        print("\nAll 50 images were classified correctly!")

if __name__ == "__main__":
    run_sanity_test()
