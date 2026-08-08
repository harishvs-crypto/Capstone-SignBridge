import os
import cv2
import numpy as np
from backend.model_utils import SignLanguageClassifier

def test_rotation_effect():
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    if classifier.model is None:
        print("Error: Model could not be loaded.")
        return

    # dataset/a/my205.jpg is predicted as A with 100% confidence
    sample_path = 'dataset/a/my205.jpg'
    if not os.path.exists(sample_path):
        print(f"Error: '{sample_path}' not found.")
        return

    img = cv2.imread(sample_path)
    
    # 0. Original (Horizontal, wrist at left, pointing right)
    pred_0, conf_0 = classifier.predict(img)
    
    # Rotate 90 degrees Counter-Clockwise (simulating vertical hand - wrist at bottom, pointing up)
    img_90_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pred_90_ccw, conf_90_ccw = classifier.predict(img_90_ccw)
    
    # Rotate 180 degrees
    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    pred_180, conf_180 = classifier.predict(img_180)
    
    # Rotate 90 degrees Clockwise
    img_90_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    pred_90_cw, conf_90_cw = classifier.predict(img_90_cw)

    print("=== ROTATION TEST ON CORRECT IMAGE (my205.jpg) ===")
    print(f"Original (Horizontal, Right-pointing): Predicted {pred_0:<5} | Confidence {conf_0*100:6.2f}%")
    print(f"Rotated 90 CCW (Vertical, Up-pointing): Predicted {pred_90_ccw:<5} | Confidence {conf_90_ccw*100:6.2f}%")
    print(f"Rotated 180 (Horizontal, Left-pointing): Predicted {pred_180:<5} | Confidence {conf_180*100:6.2f}%")
    print(f"Rotated 90 CW (Vertical, Down-pointing): Predicted {pred_90_cw:<5} | Confidence {conf_90_cw*100:6.2f}%")

if __name__ == "__main__":
    test_rotation_effect()
