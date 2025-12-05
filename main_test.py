import cv2
import matplotlib.pyplot as plt
from src.hog_implementation import compute_gradients, visualize_hog

img = cv2.imread("data/test_images/sample.png", 0)

# 1) Gradients
mag, ang = compute_gradients(img)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(mag, cmap="gray")
plt.title("Gradient Magnitude")

plt.subplot(1,2,2)
plt.imshow(ang, cmap="gray")
plt.title("Gradient Orientation")
plt.savefig("outputs/hog_test_results/gradients_single.png")
plt.close()

# 2) HOG (8x8 - 9 bin)
hog1 = visualize_hog(img, cell_size=(8,8), num_bins=9)
cv2.imwrite("outputs/hog_test_results/hog_8x8_9bins.png", hog1)

# 3) HOG (16x16 - 9 bin)
hog2 = visualize_hog(img, cell_size=(16,16), num_bins=9)
cv2.imwrite("outputs/hog_test_results/hog_16x16_9bins.png", hog2)

# 4) HOG (8x8 - 18 bin)
hog3 = visualize_hog(img, cell_size=(8,8), num_bins=18)
cv2.imwrite("outputs/hog_test_results/hog_8x8_18bins.png", hog3)

print("✔ main_test.py bitti – tüm görseller kaydedildi!")
