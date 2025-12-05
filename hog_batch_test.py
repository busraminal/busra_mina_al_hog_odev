import glob
import os
import cv2
import numpy as np
from src.hog_implementation import compute_gradients, compute_hog_descriptor, visualize_hog

# =============================
#  ÇIKTI KLASÖRÜ
# =============================
output_dir = "outputs/hog_test_results"
os.makedirs(output_dir, exist_ok=True)

# =============================
#  GÖRSELLERİ YÜKLE
# =============================
image_paths = glob.glob("data/hog_test/*")
print("DEBUG found:", image_paths)

if len(image_paths) < 5:
    print("[UYARI] En az 5 görüntü eklemelisin: data/hog_test/")
    exit()

# =============================
#  TÜM PARAMETRELER
# =============================
hog_settings = [
    ("hog_8x8_9bins",  (8, 8),  9),
    ("hog_16x16_9bins", (16, 16), 9),
    ("hog_8x8_18bins",  (8, 8), 18),
]

# =============================
#  TÜM GÖRSELLERİ İŞLE
# =============================
for img_path in image_paths:
    print(f"[INFO] İşleniyor: {img_path}")

    img = cv2.imread(img_path, 0)
    if img is None:
        print("[HATA] Görüntü okunamadı:", img_path)
        continue

    base = os.path.basename(img_path).replace(".png", "")

    # ============================================
    # 1) GRADIENT MAGNITUDE & ORIENTATION
    # ============================================
    mag, ang = compute_gradients(img)

    cv2.imwrite(f"{output_dir}/{base}_gradient_mag.png", mag)
    cv2.imwrite(f"{output_dir}/{base}_gradient_ang.png", ang)

    # ============================================
    # 2) HOG PARAMETRE KARŞILAŞTIRMALARI
    # ============================================
    for name, cell, bins in hog_settings:
        hog_vis = visualize_hog(img, cell_size=cell, num_bins=bins, show=False)

        out_path = f"{output_dir}/{base}_{name}.png"
        cv2.imwrite(out_path, hog_vis)

        print(f"  [KAYDEDİLDİ] {out_path}")

print("\n✔ [TAMAMLANDI] Tüm gradient + HOG varyasyonları kaydedildi!")
