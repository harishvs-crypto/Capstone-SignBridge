import os
import cv2
import numpy as np
from backend.model_utils import SignLanguageClassifier

def test_border_effect():
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    if classifier.model is None:
        print("Error: Model could not be loaded.")
        return

    sample_path = 'dataset/a/0.jpg'
    if not os.path.exists(sample_path):
        print(f"Error: '{sample_path}' not found.")
        return

    img = cv2.imread(sample_path)
    
    # 1. Prediction without border modification
    pred_orig, conf_orig = classifier.predict(img)
    print("=== BORDER EFFECT TEST (dataset/a/0.jpg) ===")
    print(f"Original Prediction: {pred_orig:<5} | Confidence: {conf_orig*100:6.2f}%")
    
    # 2. Add white border to top and left (e.g. intensity 230, width 3 pixels)
    img_modified = img.copy()
    img_modified[0:3, :, :] = 232  # Top border
    img_modified[:, 0:3, :] = 232  # Left border
    
    pred_mod, conf_mod = classifier.predict(img_modified)
    print(f"Modified Prediction: {pred_mod:<5} | Confidence: {conf_mod*100:6.2f}%")

if __name__ == "__main__":
    test_border_effect()
