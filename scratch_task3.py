import os
import time
import tensorflow as tf
from backend.model_utils import SignLanguageClassifier

def verify_model_loading():
    classifier = SignLanguageClassifier()
    model_path = classifier.model_path
    
    abs_path = os.path.abspath(model_path)
    exists = os.path.exists(model_path)
    
    print("=== MODEL LOADING DIAGNOSIS ===")
    print(f"Default model path in class: {model_path}")
    print(f"Absolute file path:         {abs_path}")
    print(f"File exists?:               {exists}")
    
    if exists:
        mtime = os.path.getmtime(model_path)
        print(f"Last modified timestamp:    {time.ctime(mtime)} ({mtime})")
        print(f"File size (bytes):          {os.path.getsize(model_path)}")
    else:
        print("WARNING: Model file does not exist at path!")
        
    if classifier.model is not None:
        print("\n=== MODEL SUMMARY ===")
        classifier.model.summary()
    else:
        print("WARNING: Classifier model is None!")

if __name__ == "__main__":
    verify_model_loading()
