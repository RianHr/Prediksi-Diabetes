# 📊 Prediksi Diabetes Menggunakan Regresi Logistik

## 🎯 Tujuan Proyek

1. Membangun model prediksi diabetes berdasarkan data medis.
2. Menguji akurasi model menggunakan algoritma machine learning (regresi logistik).

---

## 🧠 Teknologi yang Digunakan

* Python 3
* Scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn
* Joblib

---

## 🗂️ Struktur Folder

```
prediksi_diabetes/
├── data/                    # Dataset diabetes.csv
├── model/                   # File model & scaler (.pkl)
├── hasil/                   # Hasil evaluasi dan prediksi disimpan di sini
├── visualisasi/             # Grafik evaluasi & fitur
├── src/                     # Semua modul Python utama
│   ├── muat_data.py
│   ├── preprocessing.py
│   ├── eksplorasi_data.py
│   ├── pelatihan_model.py
│   ├── evaluasi_model.py
│   ├── visualisasi.py
│   ├── prediksi.py
│   └── menu.py
├── main.py                  # Titik masuk program
├── README.md
└── requirements.txt         # Daftar pustaka Python
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone repositori dan masuk ke folder:

```bash
git clone <repo-url>
cd prediksi_diabetes
```

### 2. Install semua pustaka:

```bash
pip install -r requirements.txt
```

### 3. Jalankan program:

```bash
python main.py
```

---

## 📋 Menu Utama Aplikasi

| No | Menu                              | Deskripsi                                  |
| -- | --------------------------------- | ------------------------------------------ |
| 1  | Prediksi Diabetes untuk Data Baru | Input data pasien, prediksi & simpan hasil |
| 2  | Lihat Evaluasi Model              | Akurasi, ROC AUC, Confusion Matrix         |
| 3  | Lihat Visualisasi                 | Tampilkan grafik evaluasi dan distribusi   |
| 4  | Informasi Dataset                 | Statistik, distribusi kelas                |
| 5  | Lihat Hasil Prediksi Sebelumnya   | Menampilkan hasil prediksi terbaru         |
| 6  | Keluar                            | Keluar dari aplikasi                       |

---

## 📈 Hasil yang Dihasilkan

* `hasil/laporan_evaluasi.txt` → evaluasi model
* `hasil/hasil_prediksi.csv` → prediksi dan fitur dominan
* `visualisasi/*.png` → grafik evaluasi dan fitur

---

## 📚 Dataset

Dataset yang digunakan: [Pima Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## ✍️ Penulis

Nama: *Rian Hadi Rhamdani]*
Tahun: 2025

---

## 📌 Lisensi

Proyek ini dapat digunakan untuk pembelajaran dan penelitian.
Harap beri atribusi jika digunakan untuk keperluan akademik atau publikasi.
