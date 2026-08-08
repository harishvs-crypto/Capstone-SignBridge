import os
import json
from backend.model_utils import SignLanguageClassifier

def verify_mapping():
    # 1. Get Folder names sorted alphabetically (how Keras sorts them)
    dataset_dir = 'dataset'
    folders = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    keras_mapping = {}
    for idx, folder in enumerate(folders):
        display = 'SPACE' if folder == '{' else folder.upper()
        keras_mapping[idx] = display

    # 2. Get saved json mapping
    json_mapping = {}
    json_path = 'models/class_labels.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                json_mapping[int(k)] = v

    # 3. Get active classifier inference mapping
    classifier = SignLanguageClassifier(model_path='models/signbridge_best.h5')
    inference_mapping = {idx: name for idx, name in enumerate(classifier.classes)}

    print("=== CLASS MAPPINGS VERIFICATION ===")
    print(f"{'Index':<6} | {'Keras Directory Sort':<20} | {'class_labels.json':<18} | {'Inference Class List':<20}")
    print("-" * 75)
    
    mismatch_detected = False
    all_indices = sorted(list(set(list(keras_mapping.keys()) + list(json_mapping.keys()) + list(inference_mapping.keys()))))
    
    for idx in all_indices:
        k_val = keras_mapping.get(idx, 'N/A')
        j_val = json_mapping.get(idx, 'N/A')
        i_val = inference_mapping.get(idx, 'N/A')
        
        match = "Match"
        if k_val != j_val or j_val != i_val:
            match = "MISMATCH!"
            mismatch_detected = True
            
        print(f"{idx:<6} | {k_val:<20} | {j_val:<18} | {i_val:<20} | {match}")
        
    print("-" * 75)
    if mismatch_detected:
        print("[WARNING] Mismatches detected in mappings!")
    else:
        print("[OK] All mappings match perfectly!")

if __name__ == "__main__":
    verify_mapping()
