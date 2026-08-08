import cv2
import numpy as np

img = cv2.imread("dataset/b/0.jpg")
if img is not None:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Find all non-black pixels
    non_black = img > 0
    non_black_mask = np.any(non_black, axis=-1)
    
    pixels = ycrcb[non_black_mask]
    if len(pixels) > 0:
        mean_y = np.mean(pixels[:, 0])
        mean_cr = np.mean(pixels[:, 1])
        mean_cb = np.mean(pixels[:, 2])
        min_cr = np.min(pixels[:, 1])
        max_cr = np.max(pixels[:, 1])
        min_cb = np.min(pixels[:, 2])
        max_cb = np.max(pixels[:, 2])
        
        print("YCrCb stats for non-black pixels:")
        print(f"  Mean Y: {mean_y:.1f}, Cr: {mean_cr:.1f}, CB: {mean_cb:.1f}")
        print(f"  Cr range: {min_cr} to {max_cr}")
        print(f"  Cb range: {min_cb} to {max_cb}")
    else:
        print("No non-black pixels found.")
