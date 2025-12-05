import cv2
import numpy as np
import os
import glob

# ==========================================
# 1) Sliding Window (64x128 — HOG uyumlu)
# ==========================================
def sliding_window(img, step=16, window_size=(64, 128)):
    winW, winH = window_size
    H, W = img.shape

    for y in range(0, H - winH, step):
        for x in range(0, W - winW, step):
            patch = img[y:y+winH, x:x+winW]
            yield x, y, patch


# ==========================================
# 2) NMS — Non-Max Suppression
# ==========================================
def non_max_suppression(boxes, scores, iou_thresh=0.4):
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area1 = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area2 = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])

        iou = inter / (area1 + area2 - inter + 1e-6)

        order = order[1:][iou < iou_thresh]

    return boxes[keep], scores[keep]


# ==========================================
# 3) CAR DETECTOR (Fake Classifier)
# ==========================================
def detect_cars(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERR] Image not found:", image_path)
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor()

    boxes = []
    scores = []

    # 64x128 PENCERE ile kaydırma
    for x, y, patch in sliding_window(gray, step=16, window_size=(64, 128)):

        # TAM OTURMUYORSA GEÇ
        if patch.shape != (128, 64):
            continue

        # Güvenli HOG hesaplama
        hog_vec = hog.compute(patch)
        if hog_vec is None:
            continue

        score = float(np.mean(hog_vec))  # fake car score

        if score > 5:
            boxes.append([x, y, 64, 128])
            scores.append(score)

    # NMS
    nms_boxes, nms_scores = non_max_suppression(boxes, scores)

    raw = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(raw, (x,y), (x+w,y+h), (0,0,255), 1)

    nms_img = img.copy()
    for (x, y, w, h), s in zip(nms_boxes, nms_scores):
        cv2.rectangle(nms_img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(nms_img, f"{s:.2f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return raw, nms_img


# ==========================================
# 4) TEST RUNNER
# ==========================================
if __name__ == "__main__":
    os.makedirs("outputs/car_detections", exist_ok=True)

    for img_path in glob.glob("data/test_images/cars/*"):
        print("[INFO] İşleniyor:", img_path)

        raw, nms = detect_cars(img_path)

        if raw is None:
            continue

        base = os.path.basename(img_path)
        cv2.imwrite(f"outputs/car_detections/{base}_raw.png", raw)
        cv2.imwrite(f"outputs/car_detections/{base}_nms.png", nms)

    print("✔ Car detection tamamlandı!")
