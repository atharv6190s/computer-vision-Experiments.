import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


# ============================================================
# EXPERIMENT NO. 6
# IMAGE SEGMENTATION TECHNIQUES
# ============================================================

IMAGE_PATH = "/workspaces/computer-vision-Experiments./EXP_06_OG_IMAGE.jpg"
OUTPUT_DIR = "output_exp6"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# STEP 1: LOAD ORIGINAL IMAGE
# WORD OUTPUT: FIGURE 1 - ORIGINAL IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        "Image not found. Make sure image.jpg is in the project folder."
    )

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "01_original_image.png"),
    image
)


# ============================================================
# STEP 2: GRAYSCALE CONVERSION AND GAUSSIAN BLUR
# WORD OUTPUT: FIGURE 2 - GRAYSCALE IMAGE AFTER GAUSSIAN BLUR
# ============================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "02_grayscale_blurred.png"),
    blurred
)


# ============================================================
# STEP 3: GLOBAL THRESHOLDING
# WORD OUTPUT: FIGURE 3 - GLOBAL THRESHOLDING RESULT
# ============================================================

_, global_threshold = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "03_global_threshold.png"),
    global_threshold
)


# ============================================================
# STEP 4: OTSU'S THRESHOLDING
# WORD OUTPUT: FIGURE 4 - OTSU'S THRESHOLDING RESULT
# ============================================================

otsu_value, otsu_threshold = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "04_otsu_threshold.png"),
    otsu_threshold
)


# ============================================================
# STEP 5: ADAPTIVE THRESHOLDING
# WORD OUTPUT: FIGURE 5 - ADAPTIVE THRESHOLDING RESULT
# ============================================================

adaptive_threshold = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "05_adaptive_threshold.png"),
    adaptive_threshold
)


# ============================================================
# STEP 6: WATERSHED SEGMENTATION
# WORD OUTPUT: FIGURE 6 - WATERSHED SEGMENTATION RESULT
# ============================================================

_, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones(
    (3, 3),
    np.uint8
)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

sure_background = cv2.dilate(
    opening,
    kernel,
    iterations=3
)

dist_transform = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

_, sure_foreground = cv2.threshold(
    dist_transform,
    0.5 * dist_transform.max(),
    255,
    0
)

sure_foreground = np.uint8(
    sure_foreground
)

unknown = cv2.subtract(
    sure_background,
    sure_foreground
)

num_labels, markers = cv2.connectedComponents(
    sure_foreground
)

markers = markers + 1

markers[unknown == 255] = 0

watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)

watershed_result = image.copy()

watershed_result[markers == -1] = [
    0,
    0,
    255
]

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "06_watershed_segmentation.png"
    ),
    watershed_result
)


# ============================================================
# STEP 7: K-MEANS CLUSTERING
# WORD OUTPUT: FIGURE 7 - K-MEANS SEGMENTATION RESULT
# ============================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

pixels = image_rgb.reshape(
    (-1, 3)
)

sample_size = min(
    50000,
    len(pixels)
)

np.random.seed(42)

indices = np.random.choice(
    len(pixels),
    sample_size,
    replace=False
)

sample_pixels = pixels[
    indices
]

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(
    sample_pixels
)

labels = kmeans.predict(
    pixels
)

centers = np.uint8(
    kmeans.cluster_centers_
)

segmented_pixels = centers[
    labels
]

kmeans_result = segmented_pixels.reshape(
    image_rgb.shape
)

kmeans_bgr = cv2.cvtColor(
    kmeans_result,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "07_kmeans_segmentation.png"
    ),
    kmeans_bgr
)


# ============================================================
# STEP 8: COMPARATIVE VISUALIZATION
# WORD OUTPUT: FIGURE 8 - COMPARATIVE ANALYSIS
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.subplot(2, 3, 1)
plt.imshow(
    cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(
    global_threshold,
    cmap="gray"
)
plt.title("Global Thresholding")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(
    otsu_threshold,
    cmap="gray"
)
plt.title("Otsu Thresholding")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(
    adaptive_threshold,
    cmap="gray"
)
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(
    cv2.cvtColor(
        watershed_result,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Watershed")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(
    kmeans_result
)
plt.title("K-Means")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_segmentation_comparison.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 9: SAVE NUMERICAL K-MEANS DATA
# NOTE: WORD MEIN INKI IMAGE NAHI LAGANI
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_labels.npy"
    ),
    labels
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_centers.npy"
    ),
    centers
)


# ============================================================
# STEP 10: DISPLAY RESULTS IN TERMINAL
# ============================================================

print("=" * 60)
print("EXPERIMENT NO. 6")
print("IMAGE SEGMENTATION TECHNIQUES")
print("=" * 60)

print("\nImage Shape:", image.shape)

print(
    "Global Threshold Value:",
    127
)

print(
    "Otsu Optimal Threshold Value:",
    otsu_value
)

print(
    "K-Means Number of Clusters:",
    3
)

print(
    "Watershed Regions:",
    num_labels - 1
)

print("\nSegmentation techniques implemented:")

print("1. Global Thresholding")
print("2. Otsu's Thresholding")
print("3. Adaptive Thresholding")
print("4. Watershed Segmentation")
print("5. K-Means Clustering")

print("\nOutput files saved in:", OUTPUT_DIR)

print("\nExperiment completed successfully.")