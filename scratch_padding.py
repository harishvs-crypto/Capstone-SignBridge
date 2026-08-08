import os
import cv2
import numpy as np
from backend.model_utils import SignLanguageClassifier

def test_crop_sensitivity():
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    if classifier.model is None:
        print("Error: Model could not be loaded.")
        return

    # Let's take dataset/a/4.jpg which predicted A at 64.54%
    sample_path = 'dataset/a/4.jpg'
    if not os.path.exists(sample_path):
        print(f"Error: '{sample_path}' not found.")
        return

    img = cv2.imread(sample_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find bounding box of non-zero (hand) pixels
    non_zero = np.argwhere(gray > 0)
    if len(non_zero) == 0:
        print("Error: No hand pixels found.")
        return
        
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    
    hand_crop = img[y_min:y_max+1, x_min:x_max+1]
    h, w, _ = hand_crop.shape
    print(f"Hand bounding box size: {w}x{h}")
    
    # Try different padding percentages
    paddings = [0.10, 0.18, 0.30, 0.40, 0.50, 0.60]
    
    print("\n=== PADDING SENSITIVITY TEST (dataset/a/4.jpg) ===")
    for pad_frac in paddings:
        pad_size = int(max(w, h) * pad_frac)
        
        # Pad with black pixels
        padded = cv2.copyMakeBorder(
            hand_crop, pad_size, pad_size, pad_size, pad_size, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Make square
        h_p, w_p, _ = padded.shape
        if h_p > w_p:
            diff = h_p - w_p
            padded = cv2.copyMakeBorder(padded, 0, 0, diff//2, diff - diff//2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif w_p > h_p:
            diff = w_p - h_p
            padded = cv2.copyMakeBorder(padded, diff//2, diff - diff//2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
        pred, conf = classifier.predict(padded)
        print(f"Padding: {pad_frac*100:2.0f}% | Image Size: {padded.shape[0]}x{padded.shape[1]} | Predicted: {pred:<5} | Confidence: {conf*100:6.2f}%")

if __name__ == "__main__":
    test_crop_sensitivity()
