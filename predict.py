import sys
import os
import cv2
from backend.model_utils import SignLanguageClassifier

def predict_image(image_path, model_path='models/signbridge_best.h5'):
    """
    Loads model and predicts the sign class for a given image file path.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return None, 0.0

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at '{image_path}'.")
        return None, 0.0

    # Initialize classifier
    classifier = SignLanguageClassifier(model_path=model_path)
    
    # Check if model loaded successfully
    if classifier.model is None:
        print("Error: Could not initialize model. Have you run 'train.py' yet?")
        return None, 0.0

    # Predict
    label, confidence = classifier.predict(image)
    return label, confidence

if __name__ == "__main__":
    # If run as CLI tool
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image> [path_to_model]")
        print("\nExample:")
        print("  python predict.py dataset/A/synthetic_0.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/signbridge_cnn.h5'

    print(f"Predicting image: {image_path} using model: {model_path}")
    label, confidence = predict_image(image_path, model_path)
    
    if label:
        print("\n--- PREDICTION RESULT ---")
        print(f"Predicted Class: {label}")
        print(f"Confidence:     {confidence:.4f} ({confidence * 100:.2f}%)")
        print("-------------------------")
