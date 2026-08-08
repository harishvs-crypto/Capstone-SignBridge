import os
import cv2
import numpy as np
from backend.model_utils import SignLanguageClassifier

def evaluate_kaggle_subset():
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    if classifier.model is None:
        print("Error: Model could not be loaded.")
        return

    dataset_dir = 'dataset'
    classes = sorted(os.listdir(dataset_dir))
    
    total_kaggle = 0
    correct_kaggle = 0
    
    class_accuracies = {}
    
    print("=== EVALUATING KAGGLE SUBSET ACCURACY ===")
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        # Kaggle files do not contain 'my' in the name
        kaggle_files = [f for f in os.listdir(cls_dir) if 'my' not in f.lower() and f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not kaggle_files:
            continue
            
        cls_total = 0
        cls_correct = 0
        
        for f in kaggle_files:
            path = os.path.join(cls_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue
                
            pred_letter, _ = classifier.predict(img)
            # Normalize target label
            target = 'SPACE' if cls == '{' else cls.upper()
            
            cls_total += 1
            if pred_letter == target:
                cls_correct += 1
                
        if cls_total > 0:
            acc = cls_correct / cls_total
            class_accuracies[cls] = (cls_correct, cls_total, acc)
            correct_kaggle += cls_correct
            total_kaggle += cls_total
            print(f"Class {cls.upper():<6} | Correct: {cls_correct:3d}/{cls_total:3d} | Accuracy: {acc*100:6.2f}%")
            
    print("=" * 50)
    print(f"Overall Kaggle Subset Accuracy: {correct_kaggle}/{total_kaggle} ({correct_kaggle/total_kaggle*100:.2f}%)")

if __name__ == "__main__":
    evaluate_kaggle_subset()
