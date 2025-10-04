import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(F:\marwadi university\DSIP\")

# Exact filenames present
names = [
    "smoothening data.png",
    "smoothening data2.png",
    "smoothening data3.png",
    "smoothening data4.png",
    "smoothening data5.png",
]

# Collect only files that actually exist and can be read
images, used = [], []
missing = []
for name in names:
    p = base / name
    if not p.exists():
        missing.append(str(p))
        continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        missing.append(str(p))
        continue
    images.append(img)
    used.append(name)

if not images:
    raise FileNotFoundError("No readable images found in /home/alex/Downloads matching the expected names.")

if missing:
    print("Warning: could not read these paths:")
    for m in missing:
        print(" -", m)

def denoise(img):
    m = cv2.medianBlur(img, 5)
    b = cv2.bilateralFilter(img, 9, 75, 75)
    g = cv2.GaussianBlur(img, (5,5), 1.0)
    return m, b, g

rows = len(images)
plt.figure(figsize=(12, 2 + 2*rows))
for i, img in enumerate(images):
    m, b, g = denoise(img)
    plt.subplot(rows,4,4*i+1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(rows,4,4*i+2); plt.imshow(m, cmap='gray'); plt.title('Median 5x5'); plt.axis('off')
    plt.subplot(rows,4,4*i+3); plt.imshow(b, cmap='gray'); plt.title('Bilateral'); plt.axis('off')
    plt.subplot(rows,4,4*i+4); plt.imshow(g, cmap='gray'); plt.title('Gaussian'); plt.axis('off')

plt.tight_layout()
plt.show()