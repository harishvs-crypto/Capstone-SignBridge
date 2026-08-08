import os
import cv2
import numpy as np
from backend.hand_detector import HandDetector

def extract_landmarks():
    detector = HandDetector()
    
    samples = {
        "Custom A": "dataset/a/my205.jpg",
        "Kaggle A (pred E)": "dataset/a/0.jpg",
        "Custom E": "dataset/e/my152.jpg"
    }
    
    print("=== HAND LANDMARKS EXTRACTION ===")
    for label, path in samples.items():
        if not os.path.exists(path):
            print(f"{label}: File '{path}' not found.")
            continue
            
        img = cv2.imread(path)
        
        # Run hand landmarks detection
        results = detector.find_hand_landmarks(img)
        
        # Check if landmarks detected
        if results and hasattr(results, 'hand_landmarks') and results.hand_landmarks:
            print(f"\n{label} ({path}):")
            print("   Landmarks detected successfully!")
            
            # Wrist: landmark 0
            # Thumb tip: landmark 4
            # Index tip: landmark 8
            # Middle tip: landmark 12
            # Ring tip: landmark 16
            # Pinky tip: landmark 20
            
            pts = results.hand_landmarks[0]
            print(f"   Wrist  (0): ({pts[0].x*100:5.1f}, {pts[0].y*100:5.1f})")
            print(f"   Thumb  (4): ({pts[4].x*100:5.1f}, {pts[4].y*100:5.1f})")
            print(f"   Index  (8): ({pts[8].x*100:5.1f}, {pts[8].y*100:5.1f})")
            print(f"   Middle(12): ({pts[12].x*100:5.1f}, {pts[12].y*100:5.1f})")
            print(f"   Ring  (16): ({pts[16].x*100:5.1f}, {pts[16].y*100:5.1f})")
            print(f"   Pinky (20): ({pts[20].x*100:5.1f}, {pts[20].y*100:5.1f})")
            
            # Check relative extension: y-coordinate of tip vs MCP joint
            # MCP joints: Index=5, Middle=9, Ring=13, Pinky=17
            index_extended = pts[8].y < pts[5].y
            middle_extended = pts[12].y < pts[9].y
            ring_extended = pts[16].y < pts[13].y
            pinky_extended = pts[20].y < pts[17].y
            print(f"   Fingers Extended: Index={index_extended}, Middle={middle_extended}, Ring={ring_extended}, Pinky={pinky_extended}")
        else:
            print(f"\n{label} ({path}):")
            print("   No landmarks detected by MediaPipe.")

if __name__ == "__main__":
    extract_landmarks()
