import cv2
import numpy as np
from backend.hand_detector import HandDetector
from backend.model_utils import SignLanguageClassifier

def run_pipeline_test():
    print("=== STARTING PIPELINE TEST ===")
    
    # 1. Initialize classifier
    print("\n1. Initializing SignLanguageClassifier...")
    classifier = SignLanguageClassifier()
    print(f"   Loaded classes: {classifier.classes[:5]}... (Total: {len(classifier.classes)})")
    
    # 2. Initialize detector
    print("\n2. Initializing HandDetector...")
    detector = HandDetector()
    
    # 3. Create synthetic frame (640x480 BGR)
    print("\n3. Creating synthetic frame...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a mock white hand-like rectangle
    cv2.rectangle(frame, (300, 200), (450, 350), (200, 200, 200), -1)
    
    # 4. Run hand detection simulation
    print("\n4. Running Hand Detector...")
    results = detector.find_hand_landmarks(frame, mode="Skin Color")
    
    # Extract ROI
    cropped_roi, bbox, annotated_frame = detector.get_hand_roi(
        frame, results, mode="Skin Color"
    )
    
    if hasattr(results, 'keys'):
        print(f"   Detection results keys: {list(results.keys())}")
    else:
        print(f"   Detection results object type: {type(results)}")
        
    print(f"   Cropped ROI shape: {cropped_roi.shape if cropped_roi is not None else 'None'}")
    print(f"   Bounding Box: {bbox}")
    
    # 5. Run prediction if ROI extracted
    if cropped_roi is not None:
        print("\n5. Running Detailed Prediction on Cropped ROI...")
        pred_letter, pred_conf, top_3, probs = classifier.predict_detailed(cropped_roi)
        print(f"   Predicted Letter: {pred_letter}")
        print(f"   Confidence Score: {pred_conf:.4f} ({pred_conf*100:.2f}%)")
        print(f"   Top-3 predictions: {top_3}")
        print(f"   Probability array shape: {probs.shape if probs is not None else 'None'}")
    else:
        print("\n5. Prediction skipped because no Hand ROI was detected (synthetic frame had no real hand landmarks).")
        
    print("\n=== PIPELINE TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_pipeline_test()
