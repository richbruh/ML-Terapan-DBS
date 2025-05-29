# Laporan Proyek Predictive Analytics: Prediksi Harga Cardano melalui Time Series Forecasting - Richie Rich Kennedy Zakaria

## 1. Domain Proyek
### Latar Belakang
Cardano adalah platform smart contract Proof-of-Stake generasi ketiga sekaligus blockchain pertama yang pengembangannya ditinjau dan dikaji oleh sekelompok insinyur dan ahli kriptografi.[Pluang](https://pluang.com/blog/academy/crypto-menengah/mengenal-cardano) 


Volatilitas harga yang tinggi pada cryptocurrency termasuk Cardano membuatnya menjadi tantangan untuk investor untuk menebak pergerakan harga Cardano. Kemampuan untuk memprediksi pergerakan harga dengan akurat dapat memberikan keunggulan strategis bagi investor maupun trader dalam pengambilan keputusan. Melalui pendekatan machine learning dengan time series forecasting, proyek ini bertujuan untuk mengembangkan model prediktif yang dapat memperkirakan pergerakan harga Cardano dalam jangka pendek hingga menengah.

## 2. Business Understanding
### Problem Statement
- Bagaimana mengembangkan model prediktif yang dapat memperkirakan harga penutupan (close price) Cardano untuk 1 bulan ke depan berdasarkan data historis?

- Bagaimana memaksimalkan akurasi prediksi dengan dataset yang terbatas (hanya memiliki 500+ sampel data)?

- Bagaimana mengidentifikasi pola dan tren dalam data historis Cardano yang dapat membantu meningkatkan akurasi prediksi?

### Goals
Membangun model time series forecasting yang dapat memprediksi harga Cardano dengan error kurang dari 3% dari rata-rata harga (MAE < $0.025)
Mengidentifikasi algoritma yang paling efektif untuk prediksi harga cryptocurrency berdasarkan perbandingan beberapa pendekatan model
Menyediakan insight tentang faktor-faktor yang paling mempengaruhi pergerakan harga Cardano

### Solution Statement
1. **Prophet**: Facebook Prophet adalah pustaka sumber terbuka yang dirilis oleh tim Ilmu Data Inti Facebook . Pustaka ini tersedia dalam R dan Python . Prophet adalah prosedur untuk peramalan data deret waktu univariat (satu variabel) berdasarkan model aditif , dan implementasinya mendukung tren, musim, dan hari libur.
2. **XGBoost (Extreme Gradient Boosting)**: Algoritma berbasis ensemble tree yang dapat digunakan untuk time series forecasting dengan pendekatan supervised learning. XGBoost menawarkan kinerja yang sangat baik untuk data numerik dengan kemampuan menangani fitur non-linear, outlier, dan pola kompleks dalam data finansial. Untuk implementasi time series, XGBoost dapat dimodifikasi dengan menggunakan teknik lag features atau sliding window untuk menangkap pola temporal.
3. **LSTM**: Arsitektur deep learning yang dirancang khusus untuk mengenali pola dalam sequence data. LSTM dapat menangkap dependensi jangka panjang dan pola kompleks dalam data harga cryptocurrency yang seringkali tidak tertangkap oleh model statistik tradisional.

## 3. Data Understanding
**Sumber Data**: Sumber dan Format Data
Dataset yang digunakan dalam proyek ini merupakan data historis harga Cardano yang dikumpulkan dari Bitget [https://www.bitget.com/price/cardano/historical-data#download]. Dataset mencakup informasi harga harian dengan total  data point dalam format CSV dengan struktur sebagai berikut:

| timeOpen | timeClose | timeHigh | timeLow | priceOpen | priceHigh | priceLow | priceClose | volume |
|----------|-----------|----------|---------|-----------|-----------|----------|------------|--------|
| 1747656000000 | 1747742399999 | 1747656600000 | 1747681260000 | 0.76059904584754 | 0.76323259490082 | 0.71201474436543 | 0.74306316159313 | 870207858.87 |


Variabel Dataset
Dataset berisi 9 kolom yang menyediakan informasi lengkap tentang pergerakan harga Cardano:

| Kolom | Deskripsi |
|-------|-----------|
| **timeOpen** | Timestamp Unix (dalam milidetik) yang menandai awal periode |
| **timeClose** | Timestamp Unix (dalam milidetik) yang menandai akhir periode |
| **timeHigh** | Timestamp saat harga mencapai titik tertinggi dalam periode tersebut |
| **timeLow** | Timestamp saat harga mencapai titik terendah dalam periode tersebut |
| **priceOpen** | Harga pembukaan Cardano dalam USD pada awal periode |
| **priceHigh** | Harga tertinggi Cardano dalam USD selama periode tersebut |
| **priceLow** | Harga terendah Cardano dalam USD selama periode tersebut |
| **priceClose** | Harga penutupan Cardano dalam USD pada akhir periode (target prediksi) |
| **volume** | Volume perdagangan Cardano dalam USD selama periode tersebut |

### Eksplorasi Data Awal
Analisis awal terhadap dataset menunjukkan beberapa karakteristik penting:

- Rentang Waktu           : Dataset mencakup data harian dari periode 2018 hingga 2025 

- Distribusi Harga        : Harga Cardano menunjukkan volatilitas yang signifikan dengan range antara $0 (Rp.0,00) hingga $3.1

- Pola Musiman            : Terdapat indikasi pola musiman dalam data yang dapat dimanfaatkan untuk pemodelan time series

- Korelasi Volume-Harga   : Terdapat korelasi antara volume perdagangan dan volatilitas harga yang dapat digunakan sebagai fitur tambahan dalam prediksi

## 4. Data Preparation
1. Loading & Preprocessing Data
- Data di-load dari file CSV menggunakan pandas (pd.read_csv('cardano.csv'))
- Kolom-kolom timestamp (timeOpen, timeClose, timeHigh, timeLow) dikonversi dari Unix timestamp (milidetik) ke format datetime menggunakan pd.to_datetime()
- Dataset diurutkan berdasarkan tanggal (kolom dateOpen) menggunakan df.sort_values('dateOpen') untuk analisis time series yang akurat

2. Penanganan Missing Vlaues 
- Identifikasi missing values menggunakan df.isnull().sum()
- Penghapusan baris yang mengandung missing values dengan df.dropna()
- Verifikasi tidak ada missing values setelah pembersihan

3. Feature Engineering 
- Fitur berbasis waktu:
- day: Ekstraksi tanggal dari datetime

4. Indikator teknikal:
- MA_7, MA_30, MA_90: Moving averages dengan periode berbeda
- EMA_7, EMA_30, EMA_90: Exponential moving averages
- volatility: Selisih antara harga tertinggi dan terendah (priceHigh - priceLow)
- returnPercentage: Perubahan persentase harga harian (priceClose.pct_change() * 100)
- range_percent: Range harga dalam persentase ((priceHigh - priceLow) / priceOpen * 100)
- volume_change: Perubahan volume perdagangan (volume.pct_change())
price_change: Perubahan harga (priceClose.pct_change())
Scaling untuk Neural Networks
• Untuk model LSTM, variabel target (priceClose) di-scale menggunakan MinMaxScaler dengan rentang (0,1).
• Data hasil scaling kemudian diubah ke dalam format sequence (misalnya, window 60 hari) menggunakan fungsi create_sequences() agar model sekuensial dapat menangkap pola temporal.

5. Splitting Data
- Data dibagi menjadi training (80%) dan testing (20%)
- Training data: train_data = df.iloc[:train_size]
- Testing data: test_data = df.iloc[train_size:]
- Pemisahan dilakukan berdasarkan urutan waktu, bukan secara acak, untuk menghindari data leakage

6. Scaling Data 
- Untuk model LSTM, variabel target (priceClose) di-scale menggunakan MinMaxScaler dengan rentang (0,1)
- Scaling hanya di-fit pada data training dan diterapkan pada keduanya (training dan testing)
- Formula scaling: scaler.fit_transform() untuk training dan scaler.transform() untuk testing

6. Transformasi Data untuk Sekuensial
- Khusus untuk LSTM, data hasil scaling ditransformasi ke dalam format sequence (window 60 hari)
- Menggunakan fungsi create_sequences() untuk menghasilkan pasangan input-output sekuensial
- Setiap sequence berisi 60 hari data historis untuk memprediksi hari berikutnya

## 5. Modelling

### LSTM

#### Cara Kerja Internal LSTM

LSTM (Long Short-Term Memory) adalah arsitektur RNN khusus yang dirancang untuk mengatasi masalah vanishing gradient pada RNN standar melalui mekanisme gerbang (gates):

1. **Forget Gate**: Menentukan informasi mana dari cell state sebelumnya yang perlu dibuang. Gate ini menganalisis h_{t-1} (output sebelumnya) dan x_t (input saat ini) untuk menghasilkan nilai antara 0 dan 1 untuk setiap angka dalam cell state:
   
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   
   Nilai 0 berarti "buang sepenuhnya" dan 1 berarti "simpan sepenuhnya".

2. **Input Gate**: Menentukan nilai baru apa yang akan disimpan dalam cell state. Terdiri dari dua komponen:
   
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  // Menentukan nilai mana yang diperbarui
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  // Membuat vektor nilai kandidat baru

3. **Cell State Update**: Memperbarui cell state lama menjadi cell state baru:
   
   C_t = f_t * C_{t-1} + i_t * C̃_t
   
   Persamaan ini menggabungkan "lupa" sebagian cell state lama dan menambahkan informasi baru.

4. **Output Gate**: Menentukan output berdasarkan cell state yang telah diperbarui:
   
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)

Mekanisme gerbang inilah yang memungkinkan LSTM untuk:
- Menyimpan informasi penting untuk waktu yang lama (mengatasi vanishing gradient)
- Membuang informasi yang tidak relevan
- Memperbarui state dengan informasi baru yang relevan
- Menentukan informasi mana yang menjadi output

Dalam prediksi harga Cardano, LSTM dapat "mengingat" pola-pola penting dalam data historis seperti tren, siklus pasar, dan reaksi terhadap kejadian eksternal, yang kemudian digunakan untuk memprediksi harga masa depan.

#### Model Architecture: 
- Input Layer: LSTM dengan 50 unit, return_sequences=True, input_shape=(60, 1)
- Dropout Layer: Rate 0.2 (20%)
- Hidden Layer: LSTM dengan 50 unit, return_sequences=False
- Dropout Layer: Rate 0.2 (20%)
- Output Layer: Dense dengan 1 unit

#### Model Compilation
- Optimizer: Adam (learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
- Loss Function: Mean Squared Error

#### Training Parameters
- Epochs: 100
- Batch Size: 32
- Validation Split: 0.2 (20%)
- Early Stopping: patience=15, restore_best_weights=True, monitored val_loss

### Prophet
a. Model Parameters
- daily_seasonality: False
- weekly_seasonality: True
- yearly_seasonality: False
- changepoint_prior_scale: 0.05 (mengontrol fleksibilitas tren)
- seasonality_prior_scale: 10.0 (mengontrol kekuatan komponen musiman)

- Data khusus Prophet disiapkan dalam format DataFrame dengan kolom ds (tanggal, dari df['dateOpen']) dan y (harga, dari df['priceClose']).

- Data kemudian di-split menjadi training (80%) dan testing (20%).

- Model Prophet diinisialisasi dengan parameter untuk seasonalitas mingguan, tahunan, dan pengaturan fleksibilitas tren (changepoint prior scale) serta seasonality prior scale.

- Model di-fit pada data training, lalu digunakan fungsi make_future_dataframe() untuk membuat dataframe future dengan periode yang sama dengan data testing.

- Forecast dilakukan, dan hasil prediksi diambil dari tail forecast untuk periode testing.

### XGBoost

#### Cara Kerja Internal XGBoost

XGBoost (eXtreme Gradient Boosting) adalah implementasi lanjutan dari algoritma gradient boosting yang bekerja melalui proses sekuensial:

1. **Sequential Tree Building**: XGBoost membangun decision trees secara bertahap:
   - Tree pertama memprediski nilai dasar
   - Setiap tree berikutnya memprediksi residual (error) dari ensemble tree sebelumnya
   - Model akhir adalah penjumlahan dari semua prediksi tree: ŷ = Σ f_i(x)

2. **Optimasi Fungsi Objektif**: XGBoost mengoptimalkan fungsi objektif yang terdiri dari:
   
   Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_i)
   
   Dimana:
   - L adalah fungsi loss (seperti squared error)
   - Ω adalah term regularisasi yang mengontrol kompleksitas model
   - Fungsi ini menyeimbangkan antara akurasi prediksi dan kompleksitas model

3. **Newton Boosting**: Menggunakan pendekatan Taylor expansion hingga orde kedua:
   - Menghitung gradien (turunan pertama) untuk menentukan arah perbaikan
   - Menghitung hessian (turunan kedua) untuk menentukan besarnya langkah perbaikan

4. **Split Finding Algorithm**: Untuk setiap pemisahan pada pohon:
   - Menghitung gain score: Gain = 1/2 * [GL²/HL + GR²/HR - (GL+GR)²/(HL+HR)] - γ
   - Memilih split dengan gain tertinggi
   - GL, GR adalah gradien pada child node kiri dan kanan
   - HL, HR adalah hessian pada child node kiri dan kanan
   - γ adalah parameter regularisasi

5. **Pruning**: Pemangkasan pohon dilakukan secara bottom-up setelah pembangunan pohon untuk mengurangi overfitting.

Untuk prediksi harga Cardano, mekanisme ini memungkinkan XGBoost untuk:
- Menangkap hubungan non-linear kompleks antara harga sebelumnya dan harga masa depan
- Mengidentifikasi pola penting dalam indikator teknikal
- Meminimalkan overfitting melalui regularisasi
- Memberikan prediksi yang akurat dengan komputasi efisien

#### Model Parameters
- objective: 'reg:squarederror'
- n_estimators: 500
- learning_rate: 0.01
- max_depth: 5
- subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42

#### Training Parameters
- Model di-fit menggunakan training data dengan monitoring performa via eval_set
- Input features: 'priceOpen'
- Target variable: 'priceClose'
- Verbose mode dimatikan untuk output yang lebih bersih


## 6. Evaluation

### XGBoost
- **MAE**: 0.0265 
- **RMSE**: 0.0469  
- **Accuracy**: 95.59%

XGBoost menunjukkan error yang rendah dan tingkat akurasi yang tinggi. Hal ini mengindikasikan model ini efektif dalam menangkap pola non-linear pada data, terutama dengan penggunaan fitur seperti *priceOpen* sebagai predictor.

Note: XGBoost memberikan prediksi yang sangat akurat, hampir sepenuhnya mengikuti pola aktual termasuk lonjakan tajam



### LSTM
- **MAE**: 0.0282  
- **RMSE**: 0.0464 
- **Accuracy**: 95.34%

Model LSTM juga memiliki performa yang baik dengan metrik error yang hanya sedikit lebih tinggi dibandingkan XGBoost. Meskipun demikian, arsitektur LSTM yang dirancang untuk menangkap informasi temporal membantu model ini untuk memahami dependensi jangka panjang dalam data time series.

Note: LSTM juga menunjukkan akurasi tinggi, dengan sedikit lag pada beberapa titik perubahan tajam

### Prophet
- **MAE**: 0.2162 
- **RMSE**: 0.2901 
- **Accuracy**: 64.03%`
Prophet memberikan error yang jauh lebih tinggi dibandingkan dengan XGBoost dan LSTM. Pendekatan univariat dari Prophet, yang mengandalkan tren dan seasonalitas, tampaknya kurang mampu menangkap fluktuasi harga jangka pendek yang kompleks.

Prophet menghasilkan prediksi yang lebih halus dan kurang akurat, gagal menangkap volatilitas dan fluktuasi harga yang tajam

Note: Prophet menghasilkan prediksi yang lebih halus dan kurang akurat, gagal menangkap volatilitas dan fluktuasi harga yang tajam

**Kesimpulan**:  
Kesimpulannya, model berbasis machine learning (XGBoost) dan deep learning (LSTM) jauh lebih efektif untuk memprediksi harga aset kripto yang memiliki volatilitas tinggi dibandingkan dengan model statistik tradisional (Prophet).