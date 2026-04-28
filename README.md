# 📈 Komoditas-AI Backend System

Sistem prediksi harga bahan pokok harian yang cerdas, dirancang khusus untuk pasar Indonesia. Backend ini menggabungkan teknik statistik klasik dengan algoritma Machine Learning modern untuk menghasilkan insight pasar yang akurat dan siap saji bagi aplikasi mobile.

---

## 🏗 Arsitektur Sistem

1.  **Data Ingestion (`requests`):** Sinkronisasi otomatis dengan API Bank Indonesia (PIHPS).
2.  **Preprocessing (`pandas`):** Penanganan *missing values*, interpolasi kalender (7 hari seminggu), dan penggunaan data riwayat 1 tahun untuk menangkap tren musiman.
3.  **Triple-Hybrid AI Engine:**
    *   **ARIMA:** Unggul pada data dengan tren stabil dan linear.
    *   **ETS (Exponential Smoothing):** Unggul pada fluktuasi jangka pendek yang dinamis.
    *   **Facebook Prophet:** Unggul pada pola musiman dan lonjakan hari raya (Holiday Aware).
    *   **Selection:** Model dipilih secara dinamis berdasarkan nilai MAPE (Mean Absolute Percentage Error) terendah melalui proses backtesting harian.
4.  **Insight Generation:** Mesin NLP sederhana yang mengubah angka menjadi narasi kontekstual dalam Bahasa Indonesia.
5.  **API Layer (`FastAPI`):** Menyajikan data terenkapsulasi dalam format JSON yang dioptimalkan untuk performa mobile (30 hari riwayat).

---

## 📂 Struktur Proyek & Tanggung Jawab

| File | Fungsi Utama |
| :--- | :--- |
| `run_all.py` | **Orchestrator**. Menjalankan seluruh pipeline dari update data hingga export JSON. |
| `commodity_forecaster.py` | **Core AI**. Berisi logika matematika, pencarian parameter, dan evaluasi model hibrida. |
| `api_server.py` | **Interface**. Server REST API yang melayani data ke client (Flutter). |
| `commodity_history.json` | **Database**. Penyimpanan lokal data historis harga (Central Source of Truth). |
| `output/` | Folder hasil pemrosesan (Grafik .png & JSON per komoditas). |

---

## 🚀 Panduan Instalasi & Pengoperasian

### 1. Persiapan Lingkungan
```bash
pip3 install pandas numpy statsmodels pmdarima requests fastapi uvicorn matplotlib prophet holidays
```

### 2. Update Data & Prediksi Harian
Jalankan perintah ini untuk melakukan sinkronisasi data terbaru dan menghitung ramalan 7 hari ke depan:
```bash
python3 run_all.py
```
*Tip: Disarankan untuk dijalankan via Cron Job setiap hari setelah data pasar dirilis.*

### 3. Menjalankan Server API
```bash
python3 api_server.py
```

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
      "forecast_pct": 0.0,
      "trend": "STABIL ➡️",
      "reliability": "SANGAT TINGGI",
      "market_alert": "🚨 Lonjakan harga tajam hari ini!",
      "insight": "🚨 Lonjakan harga tajam hari ini! Namun ke depannya...",
      "chart": {
        "history": [ { "date": "2026-04-01", "price": 17000 }, ... ],
        "forecast": [ { "date": "2026-04-28", "price": 17500 }, ... ]
      }
    }
  ]
}
```

---

## 📱 Panduan Integrasi Flutter

1.  **Dual-Window Chart:**
    *   Tampilkan **30 hari history** (garis solid) untuk memberikan konteks jangka pendek.
    *   Tampilkan **7 hari forecast** (garis putus-putus) untuk melihat arah harga masa depan.
2.  **Indikator Keandalan:**
    *   Gunakan field `reliability` untuk memberi tahu user seberapa akurat prediksi tersebut (SANGAT TINGGI = MAPE < 1.5%).
3.  **Color Coding:**
    *   Warna **Hijau** untuk tren "TURUN" (kabar baik).
    *   Warna **Merah** untuk tren "NAIK" (waspada).

---

## 🛠 Tech Stack
- **Language:** Python 3.14+
- **Data Science:** Pandas, Statsmodels, Pmdarima, Prophet
- **Web Framework:** FastAPI, Uvicorn
- **Integration:** PIHPS API (Bank Indonesia) & Holidays (ID)

---
**Maintained by:** Tim Analis Data & AI
