import os
import cv2
import numpy as np
from backend.model_utils import SignLanguageClassifier

def debug_prediction():
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    if classifier.model is None:
        print("Error: Model could not be loaded.")
        return

    sample_path = 'dataset/a/0.jpg'
    if not os.path.exists(sample_path):
        print(f"Error: Sample image '{sample_path}' does not exist.")
        return

    # 1. Original Image
    img = cv2.imread(sample_path)
    print(f"1. Original Image: {sample_path}")
    print(f"   Shape: {img.shape}")
    print(f"   Min/Max pixels: {img.min()} / {img.max()}")

    # 2. Hand crop image (for dataset images, they are pre-cropped)
    # Let's save a copy of the crop
    artifacts_dir = r'C:\Users\shrii\.gemini\antigravity\brain\27577877-5c0c-4bc3-8235-389eabd30873\artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    cv2.imwrite(os.path.join(artifacts_dir, 'task5_original.jpg'), img)
    print(f"   Saved original to artifacts/task5_original.jpg")

    # 3. Processed image sent to model
    processed_tensor = classifier.preprocess_image(img)
    # The output is (1, 64, 64, 3) normalized in [0, 1]
    print(f"3. Processed Image Tensor:")
    print(f"   Shape: {processed_tensor.shape}")
    print(f"   Data type: {processed_tensor.dtype}")
    print(f"   Value range: {processed_tensor.min():.4f} to {processed_tensor.max():.4f}")
    
    # Save a visual representation (upscaled so it is visible, mapped back to 0-255)
    vis_img = (processed_tensor[0] * 255.0).astype(np.uint8)
    vis_img_large = cv2.resize(vis_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(artifacts_dir, 'task5_processed.png'), vis_img_large)
    print(f"   Saved processed image visual to artifacts/task5_processed.png")

    # 4. Predict
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
    debug_prediction()
