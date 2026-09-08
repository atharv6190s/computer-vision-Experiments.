import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Load grayscale image
file_path = "/workspaces/computer-vision-Experiments./download.jpg"   # apna image path yahan daalo
img = cv2.imread(file_path, 0)

if img is None:
    raise ValueError("Image not loaded. Check file path.")

rows, cols = img.shape

# Step 2: Compute DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Step 3: Shift zero-frequency to center
dft_shift = np.fft.fftshift(dft)

# Step 4: Magnitude Spectrum
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Step 5: Low-Pass Filter
mask = np.zeros((rows, cols, 2), np.uint8)
crow, ccol = rows//2 , cols//2
r = 30  # radius of LPF
cv2.circle(mask, (ccol, crow), r, (1,1), -1)
lpf = dft_shift * mask

# Step 6: High-Pass Filter
hpf = dft_shift * (1 - mask)

# Step 7: Inverse DFT function
def idft_reconstruct(filtered):
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    return img_back

img_lpf = idft_reconstruct(lpf)
img_hpf = idft_reconstruct(hpf)

# Step 8: Save outputs separately
os.makedirs("outputs", exist_ok=True)

cv2.imwrite("outputs/original.png", img)
cv2.imwrite("outputs/magnitude_spectrum.png", magnitude_spectrum)
cv2.imwrite("outputs/lpf.png", img_lpf)
cv2.imwrite("outputs/hpf.png", img_hpf)

print("✅ Separate image files saved in outputs/ folder:")
print(" - original.png")
print(" - magnitude_spectrum.png")
print(" - lpf.png")
print(" - hpf.png")

# Step 9: Analyze impact
print("\n--- Analysis ---")
print("Low-Pass Filter: Image becomes smoother, noise reduced, but edges blurred.")
print("High-Pass Filter: Edges and fine details enhanced, but noise may increase.")

# Step 10: Document observations
observations = """
Observations:
1. Fourier Transform converts image into frequency domain for selective filtering.
2. Low-Pass filtering suppresses high-frequency noise, improves smoothness.
3. High-Pass filtering enhances edges and fine details, useful for feature extraction.
4. Frequency domain filtering is efficient for large kernel operations compared to spatial domain.
5. Practical applications include medical imaging, satellite image enhancement, and biometric systems.
"""
print(observations)
