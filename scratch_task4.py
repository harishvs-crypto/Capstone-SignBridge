# Side-by-side comparison of training and inference preprocessing

def generate_report():
    print("=== PREPROCESSING SIDE-BY-SIDE COMPARISON ===")
    print(f"{'Feature':<25} | {'Training (train.py)':<30} | {'Inference (model_utils.py)':<30}")
    print("-" * 90)
    print(f"{'Resize Dimensions':<25} | {'(64, 64)':<30} | {'(64, 64)':<30}")
    print(f"{'Color Format Input':<25} | {'RGB (via PIL/Keras load_img)':<30} | {'BGR (from OpenCV/Webcam)':<30}")
    print(f"{'Grayscale Conversion':<25} | {'None (loads original gray image)':<30} | {'cv2.cvtColor(BGR2GRAY)':<30}")
    print(f"{'Contrast Stretching':<25} | {'None':<30} | {'Min-Max stretching to [15, 235]':<30}")
    print(f"{'Channel Replication':<25} | {'None (loaded directly as 3ch)':<30} | {'cv2.merge([gray, gray, gray])':<30}")
    print(f"{'Normalization':<25} | {'/ 255.0 (rescale=1/255)':<30} | {'/ 255.0':<30}")
    print(f"{'Channel Order Output':<25} | {'RGB (equivalent to R=G=B)':<30} | {'RGB (equivalent to R=G=B)':<30}")
    print(f"{'Tensor Shape':<25} | {'(None, 64, 64, 3)':<30} | {'(1, 64, 64, 3)':<30}")
    print("-" * 90)

if __name__ == "__main__":
    generate_report()
