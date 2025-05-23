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
Dataset yang digunakan dalam proyek ini merupakan data historis harga Cardano yang dikumpulkan dari Bitget (https://www.bitget.com/price/cardano/historical-data#download). Dataset mencakup informasi harga harian dengan total  data point dalam format CSV dengan struktur sebagai berikut:

timeOpen,timeClose,timeHigh,timeLow,priceOpen,priceHigh,priceLow,priceClose,volume
1747656000000,1747742399999,1747656600000,1747681260000,0.76059904584754,0.76323259490082,0.71201474436543,0.74306316159313,870207858.87

Variabel Dataset
Dataset berisi 9 kolom yang menyediakan informasi lengkap tentang pergerakan harga Cardano:

Kolom	    Deskripsi
timeOpen	: Timestamp Unix (dalam milidetik) yang menandai awal periode
timeClose	: Timestamp Unix (dalam milidetik) yang menandai akhir periode
timeHigh	: Timestamp saat harga mencapai titik tertinggi dalam periode tersebut
timeLow	    : Timestamp saat harga mencapai titik terendah dalam periode tersebut
priceOpen	: Harga pembukaan Cardano dalam USD pada awal periode
priceHigh	: Harga tertinggi Cardano dalam USD selama periode tersebut
priceLow	: Harga terendah Cardano dalam USD selama periode tersebut
priceClose	: Harga penutupan Cardano dalam USD pada akhir periode (target prediksi)
volume	    : Volume perdagangan Cardano dalam USD selama periode tersebut

### Eksplorasi Data Awal
Analisis awal terhadap dataset menunjukkan beberapa karakteristik penting:

- Rentang Waktu           : Dataset mencakup data harian dari periode 2018 hingga 2025 

- Distribusi Harga        : Harga Cardano menunjukkan volatilitas yang signifikan dengan range antara $0 (Rp.0,00) hingga $3.1

- Pola Musiman            : Terdapat indikasi pola musiman dalam data yang dapat dimanfaatkan untuk pemodelan time series

- Korelasi Volume-Harga   : Terdapat korelasi antara volume perdagangan dan volatilitas harga yang dapat digunakan sebagai fitur tambahan dalam prediksi

### Data Preparation
Loading & Preprocessing Data
• Data di-load dari file CSV dan kolom-kolom timestamp (timeOpen, timeClose, dll) dikonversi dari Unix timestamp ke format datetime.
• Dataset diurutkan berdasarkan tanggal (misalnya, kolom dateOpen) dan beberapa fitur tambahan (seperti moving averages, volatilitas, dan persentase return) dihitung.

Scaling untuk Neural Networks
• Untuk model LSTM, variabel target (priceClose) di-scale menggunakan MinMaxScaler dengan rentang (0,1).
• Data hasil scaling kemudian diubah ke dalam format sequence (misalnya, window 60 hari) menggunakan fungsi create_sequences() agar model sekuensial dapat menangkap pola temporal.

Splitting Data
• Data dipecah menjadi training dan testing (misalnya, 80% data awal sebagai training dan 20% sisanya sebagai testing).
• Untuk LSTM, hasil sequence (dengan dimensi 3D) dibagi menjadi X_train, X_test dan y_train, y_test.

### Modelling
## LSTM
• Model LSTM dibangun secara sequential dengan dua lapisan LSTM:
  - Lapisan pertama dengan 50 unit, return_sequences=True, diikuti oleh lapisan Dropout (0.2).
  - Lapisan kedua dengan 50 unit, return_sequences=False, juga diikuti oleh Dropout (0.2).
• Lapisan output menggunakan Dense layer (1 unit) untuk menghasilkan prediksi harga.
• Model dikompilasi menggunakan optimizer adam dan loss function mean_squared_error.
• Pelatihan dilakukan dengan early stopping dan model checkpoint agar model terbaik disimpan, dengan pembagian validation split (0.2) dan sejumlah epoch (misal 100 epoch).

## Prophet

• Data khusus Prophet disiapkan dalam format DataFrame dengan kolom ds (tanggal, dari df['dateOpen']) dan y (harga, dari df['priceClose']).
• Data kemudian di-split menjadi training (80%) dan testing (20%).
• Model Prophet diinisialisasi dengan parameter untuk seasonalitas mingguan, tahunan, dan pengaturan fleksibilitas tren (changepoint prior scale) serta seasonality prior scale.
• Model di-fit pada data training, lalu digunakan fungsi make_future_dataframe() untuk membuat dataframe future dengan periode yang sama dengan data testing.
• Forecast dilakukan, dan hasil prediksi diambil dari tail forecast untuk periode testing.

## XGBoost
• Data model XGBoost diambil dari subset training dan testing (misalnya, menggunakan fitur priceOpen untuk memprediksi priceClose).
• Model XGBoost diinisialisasi menggunakan XGBRegressor dengan parameter seperti n_estimators, learning_rate, max_depth, subsample, dan colsample_bytree.
• Model di-fit menggunakan eval_set (untuk memonitor performa pada train dan test) dan kemudian digunakan untuk melakukan prediksi pada test set.


### Evaluation

#### XGBoost
- **MAE**: 0.0261  
- **RMSE**: 0.0465  
- **Accuracy**: 95.68%

XGBoost menunjukkan error yang rendah dan tingkat akurasi yang tinggi. Hal ini mengindikasikan model ini efektif dalam menangkap pola non-linear pada data, terutama dengan penggunaan fitur seperti *priceOpen* sebagai predictor.

Note: XGBoost memberikan prediksi yang sangat akurat, hampir sepenuhnya mengikuti pola aktual termasuk lonjakan tajam



#### LSTM
- **MAE**: 0.0300  
- **RMSE**: 0.0495  
- **Accuracy**: 95.06%

Model LSTM juga memiliki performa yang baik dengan metrik error yang hanya sedikit lebih tinggi dibandingkan XGBoost. Meskipun demikian, arsitektur LSTM yang dirancang untuk menangkap informasi temporal membantu model ini untuk memahami dependensi jangka panjang dalam data time series.

Note: LSTM juga menunjukkan akurasi tinggi, dengan sedikit lag pada beberapa titik perubahan tajam

#### Prophet
- **MAE**: 0.1700  
- **RMSE**: 0.2360  
- **Accuracy**: 71.89%

Prophet memberikan error yang jauh lebih tinggi dibandingkan dengan XGBoost dan LSTM. Pendekatan univariat dari Prophet, yang mengandalkan tren dan seasonalitas, tampaknya kurang mampu menangkap fluktuasi harga jangka pendek yang kompleks.

Prophet menghasilkan prediksi yang lebih halus dan kurang akurat, gagal menangkap volatilitas dan fluktuasi harga yang tajam

Note: Prophet menghasilkan prediksi yang lebih halus dan kurang akurat, gagal menangkap volatilitas dan fluktuasi harga yang tajam

**Kesimpulan**:  
Kesimpulannya, model berbasis machine learning (XGBoost) dan deep learning (LSTM) jauh lebih efektif untuk memprediksi harga aset kripto yang memiliki volatilitas tinggi dibandingkan dengan model statistik tradisional (Prophet).