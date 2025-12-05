# 🧠 HOG Based Object Detection & Classification  
### Bilgisayarla Görü – Histogram of Oriented Gradients (HOG) Uygulaması  
**Büşra Mina AL – OSTİM Teknik Üniversitesi, AI Engineering**

---

## 📌 Proje Özeti

Bu proje, Histogram of Oriented Gradients (HOG) yöntemi kullanılarak:

- Görüntülerden özellik çıkarma  
- İnsan tespiti (pedestrian detection)  
- Araç tespiti (custom object detection)  
- HOG + SVM ile görüntü sınıflandırma  

gibi bilgisayarla görü görevlerini gerçekleştirmektedir.

---

# 🖼️ Örnek Çıktılar

Aşağıdaki görseller proje çıktılarından oluşur.  
Görselleri repo içinde şu klasöre koymalısın: **report/figures/**



### 🔹 HOG Özellik Görselleştirmesi
![HOG Visualization](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/hog_crop001036.png)

### 🔹 İnsan Tespiti (Pedestrian Detection)
![Detection Example 1](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/crop001504.png)
![Detection Example 2](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/crop001512.png)

### 🔹 Araç Tespiti (Custom Detector)
![Car Detection](https://raw.githubusercontent.com/busraminal/busra_mina_al_hog_odev/main/report/figures/person_204.png)

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

### 1) HOG Test
```bash
python src/hog_implementation.py
```

### 2) İnsan Tespiti
```bash
python src/object_detection.py
```

### 3) Araç Tespiti
```bash
python src/car_detection.py
```

### 4) Sınıflandırma
```bash
python src/classification.py
```

---

## 📂 Proje Dosya Yapısı

```
project/
├── src/
│   ├── hog_implementation.py
│   ├── object_detection.py
│   ├── classification.py
│   ├── utils.py
│   └── car_detection.py
│
├── data/
│   ├── training_set/
│   └── test_images/
│
├── outputs/
│   ├── detections/
│   ├── hog_test_results/
│   ├── car_detections/
│   └── classification_results/
│
├── models/
│   └── trained_classifier.pkl
│
├── report/
│   ├── report.pdf
│   └── figures/
│       ├── hog_vis_01.png
│       ├── detection_01.png
│       ├── detection_02.png
│       ├── car_detection_01.png
│       ├── classification_matrix.png
│       └── accuracy_plot.png
│
├── notebooks/
│   └── analysis.ipynb
│
├── README.md
└── requirements.txt
```

---

## 📦 Kullanılan Teknolojiler

- Python 3.x  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-Learn  
- Scikit-Image  
- Joblib  

---

## 📝 Sonuç

Bu projede HOG’un:
- Kenar tabanlı özellik çıkarımı  
- İnsan ve araç tespiti  
- SVM ile sınıflandırma  

gibi alanlardaki gücü test edilmiştir.  
HOG derin öğrenme yöntemlerine göre daha hafif olmakla birlikte, klasik bilgisayarla görü problemlerinde halen etkilidir.

---

## 👤 Geliştirici  
**Büşra Mina AL**  
📧 busraminaa@gmail.com  
🔗 GitHub: https://github.com/busraminal  
🔗 LinkedIn: https://www.linkedin.com/in/bmi%CC%87nal60135806/
