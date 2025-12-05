import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# 1) GRADYAN HESABI
# ============================================================
def compute_gradients(image):
    """
    Görüntü gradyan büyüklüğü (magnitude) ve yönü (angle) hesaplar.
    HOG için açı 0–180° aralığına getirilir.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)

    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(gx**2 + gy**2 + 1e-6)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180

    return magnitude, angle


# ============================================================
# 2) CELL HISTOGRAMI (Düzeltilmiş – interpolation yok ama stabil)
# ============================================================
def create_cell_histogram(cell_mag, cell_ang, num_bins=9):
    """
    Bir hücre için yönelim histogramı oluşturur.
    En büyük stabilite için doğru bin atanması sağlandı.
    """
    hist = np.zeros(num_bins, dtype=np.float32)
    bin_width = 180 / num_bins

    angles = cell_ang.flatten()
    mags   = cell_mag.flatten()

    for a, m in zip(angles, mags):
        bin_idx = int(a // bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        hist[bin_idx] += m

    return hist


# ============================================================
# 3) BLOK NORMALİZASYONU
# ============================================================
def normalize_block(block_hist, eps=1e-6):
    """
    L2 normalization (default en iyi çalışan yöntem)
    """
    norm = np.sqrt(np.sum(block_hist**2) + eps)
    return block_hist / norm


# ============================================================
# 4) HOG DESCRIPTOR HESABI
# ============================================================
def compute_hog_descriptor(image, cell_size=(8, 8), block_size=(2, 2), num_bins=9):
    """
    Tüm görüntü HOG vektörünü üretir.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mag, ang = compute_gradients(image)
    h, w = image.shape

    cell_h, cell_w = cell_size
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    # CELL HISTOGRAM MATRİSİ
    cell_hist = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w

            cell_hist[i, j] = create_cell_histogram(
                mag[y0:y1, x0:x1],
                ang[y0:y1, x0:x1],
                num_bins
            )

    # BLOCK NORMALİZASYONU
    b_y, b_x = block_size
    blocks_y = n_cells_y - b_y + 1
    blocks_x = n_cells_x - b_x + 1

    hog_vec = []

    for y in range(blocks_y):
        for x in range(blocks_x):
            block = cell_hist[y:y + b_y, x:x + b_x].ravel()
            block_norm = normalize_block(block)
            hog_vec.append(block_norm)

    return np.concatenate(hog_vec)


# ============================================================
# 5) HOG GÖRSELLEŞTİRME – PROFESYONEL ÇİZİM
# ============================================================
def visualize_hog(image, cell_size=(8,8), num_bins=9, show=True):
    """
    HOG yön çizgilerini gerçek anlamda görselleştirir.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    mag, ang = compute_gradients(gray)

    h, w = gray.shape
    cell_h, cell_w = cell_size

    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    bin_edges = np.linspace(0, 180, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    vis = np.zeros((h, w), dtype=np.float32)

    for i in range(n_cells_y):
        for j in range(n_cells_x):

            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w

            hist = create_cell_histogram(mag[y0:y1, x0:x1], ang[y0:y1, x0:x1], num_bins)
            hist /= (hist.max() + 1e-6)

            cx = j * cell_w + cell_w // 2
            cy = i * cell_h + cell_h // 2

            for b, v in enumerate(hist):
                angle = np.deg2rad(bin_centers[b])
                dx = int(np.cos(angle) * v * (cell_w / 2))
                dy = int(np.sin(angle) * v * (cell_h / 2))

                x1p, x2p = cx - dx, cx + dx
                y1p, y2p = cy - dy, cy + dy

                if 0 <= x1p < w and 0 <= x2p < w and 0 <= y1p < h and 0 <= y2p < h:
                    cv2.line(vis, (x1p, y1p), (x2p, y2p), 255, 1)

    if show:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(vis, cmap='gray')
        plt.title(f"HOG Visualization (cell={cell_size}, bins={num_bins})")
        plt.axis("off")
        plt.show()

    return vis


if __name__ == "__main__":
    import os

    print("→ HOG test başlatılıyor...")

    test_dir = "data/test_images"
    files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))]

    if not files:
        print("test_images klasörü boş!")
        exit()

    img_path = os.path.join(test_dir, files[0])
    print("→ Kullanılan görüntü:", img_path)

    img = cv2.imread(img_path)
    vis = visualize_hog(img, show=True)

    hog_vec = compute_hog_descriptor(img)
    print("HOG boyutu:", len(hog_vec))
