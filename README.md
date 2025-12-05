# 🧠 HOG Based Object Detection & Classification  
### Histogram of Oriented Gradients (HOG) ile Nesne Tespiti ve Sınıflandırma  
**Büşra Mina AL – OSTİM Teknik Üniversitesi, AI Engineering**

---

## 📌 Proje Özeti

Bu proje, geleneksel bilgisayarla görü yöntemlerinden biri olan **Histogram of Oriented Gradients (HOG)** algoritmasını kullanarak:

- Görüntüden özellik çıkarımı  
- İnsan tespiti (HOG + SVM pedestrian detector)  
- Araç tespiti (sliding window + HOG SVM)  
- HOG tabanlı görüntü sınıflandırma  

gibi görevleri gerçekleştirmektedir.

---

# 🖼️ Örnek Çıktılar

### 🔹 HOG Özellik Görselleştirmesi
![HOG Visualization](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/hog_crop001036.png)

### 🔹 İnsan Tespiti (Pedestrian Detection)
![Detection Example 1](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/crop001504.png)
![Detection Example 2](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/crop001512.png)

### 🔹 Sınıflandırma – Confusion Matrix
![Confusion Matrix](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/SVM_confusion_matrix.png)

### 🔹 Accuracy Grafiği
![Accuracy Plot](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/accuracy_comparison.png)

---

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
```

---

## ▶️ Çalıştırma Komutları

### HOG Test
```bash
python src/hog_implementation.py
```

### İnsan Tespiti
```bash
python src/object_detection.py
```

### Araç Tespiti
```bash
python src/car_detection.py
```

### Sınıflandırma
```bash
python src/classification.py
```

---

## 📂 Proje Yapısı

```
project/
├── src/
├── data/
├── outputs/
├── models/
├── report/
│   ├── report.pdf
│   ├── figures/
├── notebooks/
└── README.md
```

---

## 📝 Sonuç

Bu projede HOG’un:

- Kenar tabanlı özellik çıkarımı  
- Nesne tespiti (insan ve araç)  
- SVM ile sınıflandırma  

gibi görevlerdeki performansı incelenmiştir.

---

## 👤 Geliştirici  
**Büşra Mina AL**  
📧 busraminaa@gmail.com  
🔗 GitHub: https://github.com/busraminal  
🔗 LinkedIn: https://www.linkedin.com/in/bmi%CC%87nal60135806/
