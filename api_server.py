from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="Commodity Price API")

# Izinkan Flutter (dan semua origin) untuk mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JSON_PATH = "output/mobile_backend.json"

@app.get("/")
def home():
    return {"message": "Commodity API is Running", "status": "active"}

@app.get("/api/market-summary")
def get_market_summary():
    """Endpoint utama untuk diambil oleh Flutter."""
    if not os.path.exists(JSON_PATH):
        raise HTTPException(status_code=404, detail="Data belum digenerate. Jalankan run_all.py terlebih dahulu.")
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    import uvicorn
    print("🚀 API Server starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
