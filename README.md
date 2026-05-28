# Analisis Sentimen Review Mobile Legends: Bang Bang Menggunakan Machine Learning dan Deep Learning

## Deskripsi Proyek

Proyek ini merupakan implementasi analisis sentimen terhadap review aplikasi Mobile Legends: Bang Bang dari Google Play Store. Sistem ini menggunakan berbagai algoritma machine learning dan deep learning untuk mengklasifikasikan sentimen pengguna ke dalam tiga kategori: Positif, Netral, dan Negatif. Proyek ini dikembangkan sebagai submission untuk kursus Machine Learning Terapan di Dicoding.

Dataset diperoleh melalui web scraping secara mandiri menggunakan library google-play-scraper dengan minimal 3000 sampel data. Analisis sentimen dilakukan dengan Long Short-Term Memory (LSTM).

## Fitur Utama

- Web scraping otomatis review dari Google Play Store
- Preprocessing teks menggunakan Sastrawi (stemming dan stopword removal)
- Feature extraction menggunakan TF-IDF
- Pelabelan otomatis berdasarkan rating pengguna
- Implementasi 3 skema pelatihan berbeda:
  - Random Forest dengan TF-IDF
  - Support Vector Machine (SVM) dengan TF-IDF
  - LSTM (Deep Learning) untuk analisis sekuensial
- Evaluasi model dengan metrik akurasi, precision, recall, dan F1-score
- Inference untuk prediksi sentimen data baru

## Struktur Folder

```
submission-analisis-sentimen/
│
├── scraping_mlbb.py              # Script untuk scraping review dari Play Store
├── training_mlbb.ipynb           # Notebook pelatihan model dengan 3 skema
├── dataset_mlbb_reviews.csv      # Dataset hasil scraping
├── requirements.txt              # Dependencies Python yang dibutuhkan
└── README.md                     # Dokumentasi proyek
```

## Teknologi yang Digunakan

### Library Python
- **Data Collection**: google-play-scraper
- **Data Processing**: pandas, numpy
- **Text Processing**: Sastrawi, NLTK, re
- **Machine Learning**: scikit-learn
- **Deep Learning**: TensorFlow/Keras
- **Visualization**: matplotlib, seaborn

### Algoritma
1. **Random Forest**: Ensemble learning untuk klasifikasi dengan multiple decision trees
2. **Support Vector Machine (SVM)**: Klasifikasi dengan mencari hyperplane optimal
3. **LSTM (Long Short-Term Memory)**: Deep learning untuk memahami konteks sekuensial teks

## Instalasi

### Persyaratan Sistem
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Jupyter Notebook atau JupyterLab

### Langkah Instalasi

1. Clone atau download repository ini

2. Install dependencies yang diperlukan:
```bash
pip install -r requirements.txt
```

3. Download data NLTK (jika diperlukan):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Cara Penggunaan

### 1. Scraping Data

Jalankan script scraping untuk mendapatkan dataset review:

```bash
python scraping_mlbb.py
```

Script ini akan:
- Mengambil minimal 3000 review dari Mobile Legends: Bang Bang
- Menyimpan hasil ke file `dataset_mlbb_reviews.csv`
- Melakukan pelabelan otomatis berdasarkan rating

### 2. Training Model

Buka dan jalankan notebook pelatihan:

```bash
jupyter notebook training_mlbb.ipynb
```

Atau gunakan JupyterLab:

```bash
jupyter lab training_mlbb.ipynb
```

Notebook akan menjalankan:
- Exploratory Data Analysis (EDA)
- Text preprocessing
- Feature extraction dengan TF-IDF
- Training 3 model berbeda
- Evaluasi dan perbandingan performa
- Inference dengan data baru

### 3. Inference

Untuk memprediksi sentimen review baru, gunakan fungsi yang tersedia di notebook:

```python
predict_sentiment("Game ini sangat bagus dan seru!", model, vectorizer)
```

## Metodologi

### 1. Data Collection
- Sumber: Google Play Store
- Aplikasi: Mobile Legends: Bang Bang
- Jumlah: 3000+ review
- Bahasa: Indonesia

### 2. Preprocessing
Tahapan preprocessing yang dilakukan:
- Case folding (konversi ke lowercase)
- Cleaning (hapus URL, mention, hashtag, angka, punctuation)
- Tokenization
- Stopword removal menggunakan Sastrawi
- Stemming menggunakan Sastrawi

### 3. Labeling
Pelabelan otomatis berdasarkan rating:
- Rating 4-5: Positif
- Rating 3: Netral
- Rating 1-2: Negatif

### 4. Feature Extraction
Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah teks menjadi representasi numerik.

### 5. Model Training

#### Skema 1: Random Forest + TF-IDF + Split 80/20
- Algoritma: Random Forest Classifier
- Feature: TF-IDF
- Data split: 80% training, 20% testing

#### Skema 2: SVM + TF-IDF + Split 80/20
- Algoritma: Support Vector Machine
- Feature: TF-IDF
- Data split: 80% training, 20% testing

#### Skema 3: LSTM + Embedding + Split 70/30
- Algoritma: LSTM (Deep Learning)
- Feature: Word Embedding
- Data split: 70% training, 30% testing

### 6. Evaluation
Metrik evaluasi yang digunakan:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Hasil dan Performa

Ketiga skema pelatihan mencapai akurasi minimal 85% pada testing set, memenuhi kriteria submission. Perbandingan detail performa model dapat dilihat pada notebook pelatihan.

Target akurasi:
- Minimal: 85% (Kriteria Utama)
- Optimal: 92% (Kriteria Tambahan)

## Kriteria Submission yang Dipenuhi

### Kriteria Utama
- Kriteria 1: Data hasil scraping mandiri (3000+ sampel)
- Kriteria 2: Ekstraksi fitur (TF-IDF) dan pelabelan data
- Kriteria 3: Menggunakan algoritma machine learning dan deep learning
- Kriteria 4: Akurasi testing set minimal 85%

### Kriteria Tambahan
- Saran 3: Dataset memiliki 3 kelas (Positif, Netral, Negatif)
- Saran 5: 3 percobaan skema pelatihan berbeda
- Saran 6: Implementasi inference/testing

## Limitasi

- Dataset terbatas pada review berbahasa Indonesia
- Pelabelan otomatis berdasarkan rating mungkin tidak selalu akurat
- Model trained khusus untuk domain game Mobile Legends
- Performa dapat bervariasi untuk aplikasi atau domain lain

## Pengembangan Selanjutnya

- Ekspansi dataset dengan jumlah sampel lebih banyak
- Implementasi algoritma deep learning lainnya (BERT, Transformer)
- Fine-tuning hyperparameter untuk peningkatan akurasi
- Deployment model sebagai API atau web application
- Analisis aspek-based sentiment analysis
- Implementasi multi-label classification

## Lisensi

Proyek ini dibuat untuk keperluan pendidikan dan submission kursus Dicoding.

## Kontak

Untuk pertanyaan atau saran, silakan hubungi melalui platform Dicoding.

## Referensi

- Dicoding Indonesia - Machine Learning Terapan
- Google Play Scraper Documentation
- Scikit-learn Documentation
- TensorFlow/Keras Documentation
- Sastrawi - Indonesian Stemmer Library

---

Dibuat sebagai submission Proyek Analisis Sentimen - Dicoding Machine Learning Terapan
