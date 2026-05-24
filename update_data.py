#!/usr/bin/env python3
import os
import json
import logging
from commodity_forecaster import load_json_data, update_history_with_api

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

HISTORY_FILE = "commodity_history.json"

def main():
    logger.info("🚀 Memulai proses pembaruan data komoditas...")
    
    if not os.path.exists(HISTORY_FILE):
        logger.error(f"❌ File history tidak ditemukan: {HISTORY_FILE}")
        return

    try:
        # 1. Muat data JSON riwayat harga komoditas
        logger.info(f"📁 Memuat file data riwayat: {HISTORY_FILE}")
        df_full, _ = load_json_data(HISTORY_FILE)
        
        # 2. Hubungi API Bank Indonesia (PIHPS) untuk memperbarui data
        logger.info("🌐 Menghubungkan ke API Bank Indonesia (PIHPS) untuk sinkronisasi...")
        df_updated = update_history_with_api(df_full, HISTORY_FILE)
        
        # 3. Simpan kembali data yang telah diperbarui ke file JSON
        logger.info(f"💾 Menyimpan data yang telah diperbarui ke {HISTORY_FILE}...")
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"data": df_updated.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)
            
        logger.info("✅ Pembaruan data komoditas berhasil diselesaikan!")
        
    except Exception as e:
        logger.error(f"❌ Terjadi kesalahan saat memperbarui data: {e}", exc_info=True)

if __name__ == "__main__":
    main()
