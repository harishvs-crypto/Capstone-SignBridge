import os
import cv2
import numpy as np
import random

def analyze_dataset_hands():
    dataset_dir = 'dataset'
    classes = sorted(os.listdir(dataset_dir))
    
    ratios = []
    fill_pcts = []
    
    random.seed(42)
    
    print("=== DATASET HAND GEOMETRY ANALYSIS ===")
    print(f"{'Class':<6} | {'File':<15} | {'Hand Size':<10} | {'Aspect Ratio':<12} | {'Fill Area %':<12}")
    print("-" * 65)
    
    for cls in classes[:5]:
        cls_dir = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not files:
            continue
        # Sample 3 files per class
        samples = random.sample(files, min(3, len(files)))
        for f in samples:
            img = cv2.imread(os.path.join(cls_dir, f))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            non_zero = np.argwhere(gray > 0)
            if len(non_zero) == 0:
                continue
            
            y_min, x_min = non_zero.min(axis=0)
            y_max, x_max = non_zero.max(axis=0)
            
            w_h = x_max - x_min + 1
            h_h = y_max - y_min + 1
            aspect_ratio = w_h / h_h
            ratios.append(aspect_ratio)
            
            fill_area_pct = len(non_zero) / (img.shape[0] * img.shape[1]) * 100
            fill_pcts.append(fill_area_pct)
            
            print(f"{cls.upper():<6} | {f:<15} | {w_h}x{h_h:<5} | {aspect_ratio:<12.2f} | {fill_area_pct:<12.2f}%")
            
    print("-" * 65)
    print(f"Mean Aspect Ratio:      {np.mean(ratios):.2f}")
    print(f"Mean Hand Area Fill:    {np.mean(fill_pcts):.2f}%")

if __name__ == "__main__":
    analyze_dataset_hands()
