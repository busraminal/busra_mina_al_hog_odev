import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from hog_implementation import compute_hog_descriptor
import os

# =========================================================
# 0) TÜM RESİMLER SABİT BOYUTA GETİRİLİYOR (128×64)
# =========================================================
IMG_SIZE = (128, 64)

def load_images(paths, label):
    X, y = [], []
    for img_path in paths:
        img = cv2.imread(img_path, 0)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)  # <-- ZORUNLU FIX!

        hog_vec = compute_hog_descriptor(img)
        X.append(hog_vec)
        y.append(label)
    return X, y


# =========================================================
# OUTPUT KLASÖRÜ
# =========================================================
output_dir = "outputs/classification_results"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 1) EĞİTİM VERİSİ
# =========================================================
train_humans = glob.glob("data/classification/humans/*")
train_nonhumans = glob.glob("data/classification/nonhumans/*")

X1, y1 = load_images(train_humans, 1)
X2, y2 = load_images(train_nonhumans, 0)

X = np.array(X1 + X2)
y = np.array(y1 + y2)

print("[INFO] Eğitim verisi yüklendi:", X.shape)


# =========================================================
# 2) TEST VERİSİ
# =========================================================
test_humans = glob.glob("data/test_images/humans_test/*")
test_nonhumans = glob.glob("data/test_images/nonhumans_test/*")

X_test, y_true = [], []

# İnsanlar
for img_path in test_humans:
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, IMG_SIZE)
    X_test.append(compute_hog_descriptor(img))
    y_true.append(1)

# İnsan olmayanlar
for img_path in test_nonhumans:
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, IMG_SIZE)
    X_test.append(compute_hog_descriptor(img))
    y_true.append(0)

X_test = np.array(X_test)


# =========================================================
# 3) MODELLER
# =========================================================
models = {
    "SVM": LinearSVC(),
    "CART": DecisionTreeClassifier(max_depth=10),
    "LogReg": LogisticRegression(max_iter=300)
}

results = {}


# =========================================================
# 4) EĞİT & TEST ET
# =========================================================
for name, model in models.items():
    print(f"\n===== {name} Modeli Eğitiliyor =====")
    model.fit(X, y)
    preds = model.predict(X_test)

    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)

    results[name] = (acc, cm)

    print(f"{name} Accuracy: {acc}")
    print(f"{name} Confusion Matrix:\n{cm}")

    # Confusion Matrix Kaydet
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_confusion_matrix.png")
    plt.close()


# =========================================================
# 5) Accuracy Karşılaştırma Grafiği
# =========================================================
names = list(results.keys())
accs = [results[n][0] for n in names]

plt.figure(figsize=(6, 4))
plt.bar(names, accs, color=["blue", "green", "orange"])
plt.title("Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)

for i, v in enumerate(accs):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig(f"{output_dir}/accuracy_comparison.png")
plt.close()

print("\n✔ TAMAMLANDI → Sonuçlar outputs/classification_results içine kaydedildi.")
