"""LLM-assisted market insights with deterministic fallbacks."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ Gagal setup LLM Client: {e}")


def generate_commodity_insight(name, trend, forecast_pct, alert):
    """Menghasilkan analisis dua perspektif (Masyarakat & Pedagang) menggunakan LLM."""
    if alert:
        if "Penurunan" in alert:
            fallback_masyarakat = f"{alert} Saatnya belanja stok bulanan selagi harga melandai."
        elif "Lonjakan" in alert:
            fallback_masyarakat = f"{alert} Sebaiknya tunda pembelian besar atau cari alternatif komoditas lain."
        else:
            fallback_masyarakat = f"{alert} Pantau harga harian untuk mendapatkan waktu beli terbaik."
    else:
        if "NAIK" in trend:
            fallback_masyarakat = f"Tren {name} diprediksi naik {abs(forecast_pct)}%. Segera stok secukupnya sebelum harga makin mahal."
        elif "TURUN" in trend:
            fallback_masyarakat = f"Kabar baik, {name} sedang turun. Tunggu beberapa hari lagi untuk harga terendah."
        else:
            fallback_masyarakat = f"Harga {name} stabil. Beli sesuai kebutuhan harian Anda."

    if "NAIK" in trend:
        fallback_pedagang = "Potensi margin meningkat karena tren naik. Pastikan stok aman untuk memenuhi permintaan."
    elif "TURUN" in trend:
        fallback_pedagang = "Hati-hati stok menumpuk saat harga turun. Percepat perputaran stok agar tidak rugi."
    else:
        fallback_pedagang = "Pasar stabil. Fokus pada efisiensi operasional dan kualitas produk."

    disclaimer_text = (
        "Analisis ini dihasilkan secara otomatis oleh AI dan bersifat saran referensi. "
        "Keputusan ekonomi tetap berada di tangan pengguna."
    )

    if not openai_client:
        return {"masyarakat": fallback_masyarakat, "pedagang": fallback_pedagang, "disclaimer": disclaimer_text}

    try:
        prompt = f"""Kamu adalah analis pasar bahan pokok profesional.
Buatlah analisis singkat untuk komoditas {name} dari DUA PERSPEKTIF berbeda:
1. MASYARAKAT (Pembeli/Konsumen): Berikan saran strategis apakah harus membeli sekarang, stok barang, atau tunda pembelian.
2. PEDAGANG (Penjual): Berikan saran bisnis terkait manajemen stok, strategi harga, atau potensi keuntungan.

Data pendukung (WAJIB DIIKUTI):
- Tren AI Masa Depan: {trend} (Ini adalah prediksi utama, jangan dilawan)
- Prediksi perubahan harga: {forecast_pct}% minggu depan.
- Kondisi HARI INI: {alert if alert else 'Relatif stabil'}.
- Pertimbangkan kondisi hari ini(wajib)

Instruksi Khusus:
- Jika tren {trend} adalah NAIK, jangan menyarankan tunda pembelian dengan alasan harga akan turun.
- Jika tren {trend} adalah TURUN, jangan menyarankan beli sekarang dengan alasan harga akan naik.
- Pastikan insight MASYARAKAT dan PEDAGANG konsisten dengan data Tren AI Masa Depan di atas.

Syarat:
1. Gunakan bahasa Indonesia yang santai tapi profesional.
2. Maksimal 30 kata per perspektif.
3. JANGAN gunakan poin-poin.
4. Output WAJIB dalam format JSON murni:
{{"masyarakat": "...", "pedagang": "..."}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        if response.choices:
            result = json.loads(response.choices[0].message.content.strip())
            result["disclaimer"] = disclaimer_text
            return result
    except Exception as e:
        print(f"⚠️ Gagal menggunakan LLM untuk {name}: {e}")

    return {"masyarakat": fallback_masyarakat, "pedagang": fallback_pedagang, "disclaimer": disclaimer_text}


def generate_global_insight(summary_data):
    """Memberikan analisis pasar secara keseluruhan."""
    naik = [i["name"] for i in summary_data if "NAIK" in i["trend"]]
    turun = [i["name"] for i in summary_data if "TURUN" in i["trend"]]

    if len(naik) > 3:
        fallback_msg = "Pasar sedang mengalami tren kenaikan harga di beberapa sektor utama. "
    elif len(turun) > 3:
        fallback_msg = "Pasar menunjukkan tren penurunan harga yang cukup luas hari ini. "
    elif naik or turun:
        fallback_msg = "Terdapat fluktuasi harga pada beberapa komoditas, namun pasar secara umum masih terkendali. "
    else:
        fallback_msg = "Pasar hari ini terpantau sangat stabil tanpa perubahan harga signifikan. "

    if naik:
        fallback_msg += f"Waspadai kenaikan pada {', '.join(naik)}. "
    if turun:
        fallback_msg += f"Potensi penghematan pada {', '.join(turun)}. "

    fallback_msg += (
        "\nGunakan prediksi ini sebagai referensi pendukung, masih banyak faktor eksternal "
        "yang bisa mempengaruhi harga."
    )

    if not openai_client:
        return fallback_msg

    try:
        prompt = f"""Kamu adalah pakar ekonomi. Buatlah rangkuman pasar bahan pokok hari ini dalam maksimal 3 kalimat padat dan menarik.
Komoditas yang diprediksi NAIK: {', '.join(naik) if naik else 'Tidak ada yang signifikan'}.
Komoditas yang diprediksi TURUN: {', '.join(turun) if turun else 'Tidak ada yang signifikan'}.

Tulis dengan gaya jurnalistik yang mengalir, seolah memberikan insight cepat untuk pemilik restoran atau ibu rumah tangga.
Jangan gunakan sapaan halo. Jangan pakai poin-poin. Langsung ke intinya."""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        if response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Gagal menggunakan LLM untuk global insight: {e}")

    return fallback_msg
