# 📈 Sistem Prediksi Harga Komoditas Pangan Nasional berbasis AI (ARIMA/ETS) 

Backend cerdas berbasis Python untuk memprediksi harga komoditas pangan harian di Indonesia. Sistem ini menggabungkan teknik statistik klasik dengan algoritma Machine Learning modern untuk menghasilkan insight pasar yang akurat dan siap saji bagi aplikasi mobile.

---

## 🏗 Arsitektur Sistem

1.  **Data Ingestion (`requests`):** Sinkronisasi otomatis dengan API Bank Indonesia (PIHPS).
2.  **Preprocessing (`pandas`):** Penanganan *missing values*, interpolasi kalender (7 hari seminggu), dan windowing data 90 hari.
3.  **Hybrid AI Engine (`statsmodels`, `pmdarima`):**
    *   **ARIMA:** Menangani data dengan tren linear dan musiman yang jelas.
    *   **ETS (Error, Trend, Seasonal):** Menangani data dengan fluktuasi eksponensial.
    *   **Selection:** Model dipilih secara dinamis berdasarkan nilai MAPE (Mean Absolute Percentage Error) terendah melalui proses backtesting.
4.  **Insight Generation:** Mesin NLP sederhana yang mengubah angka menjadi narasi kontekstual.
5.  **API Layer (`FastAPI`):** Menyajikan data terenkapsulasi dalam format JSON yang dioptimalkan untuk performa mobile.

---

## 📂 Struktur Proyek & Tanggung Jawab

| File | Fungsi Utama |
| :--- | :--- |
| `run_all.py` | **Orchestrator**. Menjalankan seluruh pipeline dari update data hingga export JSON. |
| `commodity_forecaster.py` | **Core AI**. Berisi logika matematika, pencarian parameter, dan evaluasi model. |
| `api_server.py` | **Interface**. Server REST API yang melayani data ke client (Flutter). |
| `commodity_history.json` | **Database**. Penyimpanan lokal data historis harga (Central Source of Truth). |
| `output/` | Folder hasil pemrosesan (Grafik .png & JSON per komoditas). |

---

## 🚀 Panduan Instalasi & Pengoperasian

### 1. Persiapan Lingkungan
```bash
pip install pandas numpy statsmodels pmdarima requests fastapi uvicorn matplotlib
```

### 2. Update Data & Prediksi Harian
Jalankan perintah ini untuk melakukan sinkronisasi data terbaru dan menghitung ramalan 7 hari ke depan:
```bash
python3 run_all.py
```
*Tip: Disarankan untuk dijalankan via Cron Job setiap pukul 10:00 WIB setelah data pasar dirilis.*

### 3. Menjalankan Server API
```bash
python3 api_server.py
```
Akses di browser: `http://localhost:8000/api/commodities`

---

## 📡 Dokumentasi API (Endpoints)

### `GET /api/commodities`
Mendapatkan seluruh data komoditas beserta prediksi dan wawasan pasar.

**Contoh Response Structure:**
```json
{
  "metadata": {
    "updated_at": "2026-04-28 11:33:17",
    "global_analysis": "Pasar menunjukkan tren penurunan..."
  },
  "commodities": [
    {
      "name": "Beras",
      "current_price": 17500,
      "price_changes": {
        "day_1": 9.72,
        "day_7": 9.72
      },
      "reliability": "SANGAT TINGGI",
      "market_alert": "🚨 Lonjakan harga tajam hari ini!",
      "insight": "🚨 Lonjakan harga tajam hari ini! Namun ke depannya...",
      "chart": {
        "history": [ { "date": "2026-01-28", "price": 15750 }, ... ],
        "forecast": [ { "date": "2026-04-28", "price": 17500 }, ... ]
      }
    }
  ]
}
```

---

## 📱 Panduan Integrasi Flutter

Untuk performa UI yang maksimal, gunakan tips berikut:

1.  **Line Chart (`fl_chart`):**
    *   Gunakan `history` sebagai data utama (garis solid).
    *   Gunakan `forecast` sebagai data prediksi (garis putus-putus).
    *   *Catatan:* Titik terakhir history dan titik pertama forecast adalah sama (seamless connection).
2.  **Color Coding:**
    *   Gunakan warna **Hijau** jika `trend` mengandung kata "TURUN" (Berita baik bagi konsumen).
    *   Gunakan warna **Merah** jika `trend` mengandung kata "NAIK" (Waspada bagi konsumen).
3.  **Market Alerts:**
    *   Tampilkan widget `Banner` atau `Card` khusus jika `market_alert` tidak kosong untuk menarik perhatian user.

---

## 🛠 Tech Stack
- **Language:** Python 3.14+
- **Data Science:** Pandas, Statsmodels, Pmdarima
- **Web Framework:** FastAPI, Uvicorn
- **Integration:** PIHPS API (Bank Indonesia)

---
**Maintained by:** RizkyHamdana
