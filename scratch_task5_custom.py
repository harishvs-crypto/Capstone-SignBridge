import os
import cv2
import numpy as np
from backend.model_utils import SignLanguageClassifier

def debug_custom_prediction():
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    if classifier.model is None:
        print("Error: Model could not be loaded.")
        return

    sample_path = 'dataset/a/my205.jpg'
    if not os.path.exists(sample_path):
        print(f"Error: Sample image '{sample_path}' does not exist.")
        return

    # 1. Original Image
    img = cv2.imread(sample_path)
    print(f"1. Original Image: {sample_path}")
    print(f"   Shape: {img.shape}")
    print(f"   Min/Max pixels: {img.min()} / {img.max()}")

    # 2. Predict
    pred_class, confidence, top_3, raw_probs = classifier.predict_detailed(img)
    
    print("\n4. Raw Probability Vector (all 27 classes):")
    for idx, prob in enumerate(raw_probs):
        cls_name = classifier.classes[idx]
        print(f"   Class {idx:2d} ({cls_name:5s}): {prob:.6f} ({prob*100:6.2f}%)")

    print("\n5. Top 5 Predictions:")
    # Get top 5 sorted indices
    top_5_idx = np.argsort(raw_probs)[::-1][:5]
    for rank, idx in enumerate(top_5_idx, 1):
        cls_name = classifier.classes[idx]
        prob = raw_probs[idx]
        print(f"   Rank {rank}: {cls_name:5s} = {prob*100:6.2f}% (probability: {prob:.6f})")

if __name__ == "__main__":
    debug_custom_prediction()
