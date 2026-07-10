# 🎵 Spotify Popularity Prediction using Machine Learning

## 📌 Deskripsi Project

Project ini merupakan implementasi machine learning untuk memprediksi apakah sebuah lagu Spotify termasuk kategori populer atau tidak berdasarkan karakteristik audio lagu.

Model akan melakukan klasifikasi berdasarkan fitur audio seperti danceability, energy, loudness, tempo, valence, dan beberapa fitur lainnya.

Aplikasi prediksi dibangun menggunakan Streamlit sehingga pengguna dapat memasukkan karakteristik audio lagu dan mendapatkan hasil prediksi popularitas.


## 🎯 Problem Statement

Bagaimana memprediksi apakah sebuah lagu akan menjadi populer berdasarkan karakteristik audio yang dimiliki?

Tujuan dari project ini adalah membangun model klasifikasi yang mampu membedakan lagu populer dan tidak populer berdasarkan fitur audio Spotify.


## 📂 Dataset

Dataset yang digunakan:

Spotify Tracks Dataset Detailed (Kaggle)

Jumlah data awal:
- 114.000 data lagu
- 20 atribut

Setelah proses data cleaning:
- 113.550 data lagu

Target klasifikasi:
- 0 = Tidak Popular
- 1 = Popular


## 🔍 Fitur yang Digunakan

Fitur input model:

- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- duration_ms
- explicit


## 🧹 Data Preprocessing

Tahapan preprocessing yang dilakukan:

1. Pemeriksaan missing value
2. Penghapusan data duplikat
3. Pembentukan label popularitas dengan threshold popularity >= 40
4. Encoding fitur explicit
5. Standardisasi fitur menggunakan StandardScaler
6. Pembagian data training dan testing dengan rasio 80:20


## 🤖 Machine Learning Model

Beberapa algoritma yang diuji:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)


## 📊 Hasil Evaluasi Model

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 60.17% | 56.16% | 31.27% | 40.17% |
| Random Forest | 78.01% | 76.45% | 70.21% | 73.19% |
| SVM | 60.17% | 56.37% | 30.34% | 39.44% |


## 🏆 Model Terbaik

Berdasarkan hasil evaluasi, Random Forest dipilih sebagai model terbaik karena memiliki performa tertinggi dibandingkan model lainnya.

Performa:

- Accuracy: 72.02%
- Precision: 65.09%
- Recall: 70.21%
- F1 Score: 73.19%


## 🚀 Streamlit Application

## 🌐 Deployment

Aplikasi telah berhasil dilakukan deployment menggunakan Streamlit Cloud.

Link aplikasi:
https://spotify-popularity-prediction-mzu6nffyyax8qspnn9t9tr.streamlit.app/

Aplikasi memiliki beberapa fitur:

### Dashboard EDA
Menampilkan:
- informasi dataset
- distribusi popularity
- visualisasi data

### Prediksi Lagu
Pengguna dapat memasukkan:
- karakteristik audio lagu

Kemudian sistem memberikan prediksi:
- Popular
- Tidak Popular


### Evaluasi Model
Menampilkan:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


### Interpretasi Model
Menampilkan:
- model terbaik
- feature importance


## ⚙️ Cara Menjalankan Project

Install library:

```bash
pip install -r requirements.txt

Jalankan aplikasi: streamlit run app.py

📁 Struktur Folder
Spotify-ML-Project/

│── app.py
│── random_forest_model.pkl
│── scaler.pkl
│── features.pkl
│── requirements.txt
│── runtime.txt
│── README.md
│── spotify-tracks-dataset-detailed.csv

🛠️ Teknologi yang Digunakan
Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Seaborn
Streamlit

👨‍💻 Author
Machine Learning Project
Simpan.

