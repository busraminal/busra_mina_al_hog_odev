import cv2
import numpy as np
import os
import glob

# --------------------------------------
# NMS (Non-Maximum Suppression)
# --------------------------------------
def non_max_suppression(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores).reshape(-1)  # weights -> düzleştir

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep], scores[keep]


# --------------------------------------
# İnsan Tespiti + NMS Uygulama
# --------------------------------------
def detect_people(image_path):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    img = cv2.imread(image_path)
    if img is None:
        print("[HATA] Görüntü okunamadı:", image_path)
        return None, None

    orig = img.copy()

    boxes, weights = hog.detectMultiScale(
        img,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # --- NMS ÖNCESİ ÇİZİM (Kırmızı) ---
    for (box, score) in zip(boxes, weights.reshape(-1)):
        x, y, w, h = box
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(orig, f"{score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- NMS UYGULA ---
    nms_boxes, nms_scores = non_max_suppression(boxes, weights)

    # --- NMS SONRASI ÇİZİM (Yeşil) ---
    nms_img = img.copy()
    for (box, score) in zip(nms_boxes, nms_scores):
        x, y, w, h = box
        cv2.rectangle(nms_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(nms_img, f"{score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return orig, nms_img


# --------------------------------------
# Toplu Test Scripti
# --------------------------------------
if __name__ == "__main__":
    input_dir = "data/test_images/people"
    output_dir = "outputs/detections"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"[INFO] {len(image_paths)} görüntü bulundu.\n")

    for img_path in image_paths:
        raw, nms = detect_people(img_path)
        if raw is None:
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]

        cv2.imwrite(os.path.join(output_dir, f"{base}_raw.png"), raw)
        cv2.imwrite(os.path.join(output_dir, f"{base}_nms.png"), nms)

        print(f"[OK] {base} için raw ve nms çıktıları kaydedildi.")

    print("\n✔ TAMAMLANDI – outputs/detections klasörüne kaydedildi.\n")
