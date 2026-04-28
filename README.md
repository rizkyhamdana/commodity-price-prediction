# 📈 Commodity Price Forecast System (ARIMA + FastAPI)

Sistem prediksi harga komoditas pangan nasional berbasis Machine Learning yang terintegrasi dengan API Bank Indonesia (PIHPS) dan siap digunakan sebagai backend untuk aplikasi mobile Flutter.

## 🚀 Fitur Utama
- **Hybrid Data Engine**: Sinkronisasi otomatis data historis lokal dengan API Live PIHPS Bank Indonesia.
- **Auto-Model Selection**: Memilih model terbaik antara **ARIMA** dan **ETS (Exponential Smoothing)** secara otomatis berdasarkan nilai MAPE terkecil (backtesting).
- **Mobile-Ready Backend**: Menghasilkan output JSON terstruktur yang berisi:
  - Data grafik (History 30 hari + Forecast 7 hari).
  - Mapping Sub-komoditas (Level 2).
  - Persentase perubahan harga harian.
  - Insight otomatis berbasis AI.
- **REST API**: Dibangun menggunakan **FastAPI** untuk melayani permintaan data dari aplikasi Flutter dengan performa tinggi.

## 🛠️ Tech Stack
- **Language**: Python 3.14+
- **Machine Learning**: `statsmodels`, `pmdarima`, `pandas`, `numpy`
- **Backend API**: `FastAPI`, `Uvicorn`
- **Automation**: `Crontab` (Linux)

## 📂 Struktur Proyek
```text
├── run_all.py                 # Orchestrator (Data Fetcher + Forecaster)
├── api_server.py              # FastAPI Server untuk Flutter
├── commodity_arima_pipeline.py # Core Logic ML & API BI
├── Data lengkap...json        # Database historis lokal
├── output/                    # Folder hasil generate (JSON & PNG)
│   └── mobile_backend.json    # Endpoint data utama untuk Flutter
└── requirements.txt           # Daftar library dependencies
```

## ⚙️ Instalasi & Penggunaan

### 1. Persiapan Lingkungan
```bash
# Clone repository
git clone <url-repo-anda>
cd commodity-price-prediction

# Install library
python3 -m pip install -r requirements.txt
```

### 2. Menjalankan Generator Data
Script ini akan menarik data terbaru dari BI, menghitung prediksi, dan menyimpan hasilnya ke `output/mobile_backend.json`.
```bash
python3 run_all.py
```

### 3. Menjalankan API Server
Jalankan server untuk melayani data ke aplikasi Flutter.
```bash
python3 api_server.py
```
Akses di browser: `http://localhost:8000/api/market-summary`

## ☁️ Deployment (Server Linux/VPS)

### Menjalankan API di Background (PM2)
```bash
pm2 start api_server.py --interpreter python3 --name "commodity-api"
```

### Penjadwalan Update Data (Cron)
Gunakan `crontab -e` untuk menjalankan prediksi setiap jam 5 sore (Senin-Jumat):
```bash
0 17 * * 1-5 cd /path/ke/project && /usr/bin/python3 run_all.py >> cron.log 2>&1
```

## 📝 Catatan untuk Developer Flutter
- **Endpoint**: `GET /api/market-summary`
- **Chart Data**: Gunakan field `chart.history` (biru) dan `chart.forecast` (kuning/putus-putus) untuk library `fl_chart`.
- **Assets**: Map field `image_asset` dengan folder `assets/images/` di project Flutter Anda.

---
**Disclaimer**: Prediksi harga ini didasarkan pada data historis. Faktor eksternal seperti cuaca, kebijakan pemerintah, dan hari raya dapat mempengaruhi harga aktual di pasar.
