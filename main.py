from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import random
import math
from datetime import datetime
import os

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

app = FastAPI(title="FloodGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

warnings_history: list[dict] = []


class LocationRequest(BaseModel):
    lat: float
    lon: float
    city: Optional[str] = None


class RiskResponse(BaseModel):
    risk_score: int
    risk_level: str
    risk_color: str
    analysis: str
    recommendations: list[str]
    nearest_river: str
    flood_zone: str
    weather: dict
    timestamp: str
    source: str


def get_risk_level(score: int) -> tuple[str, str]:
    if score <= 25:   return "LOW",      "#27500A"
    if score <= 50:   return "MEDIUM",   "#854F0B"
    if score <= 75:   return "HIGH",     "#633806"
    return                   "CRITICAL", "#A32D2D"


def get_risk_color_bg(score: int) -> str:
    if score <= 25:  return "#EAF3DE"
    if score <= 50:  return "#FAEEDA"
    if score <= 75:  return "#FAEEDA"
    return                  "#FCEBEB"


def geo_risk_profile(lat: float, lon: float) -> dict:
    import math
    hotspots = [
        {"lat": -6.18, "lon": 106.83, "r": 0.7, "risk": 0.90, "label": "Jakarta Utara/Barat"},
        {"lat": -6.21, "lon": 106.84, "r": 0.5, "risk": 0.80, "label": "Jakarta Pusat/Timur"},
        {"lat": -6.24, "lon": 107.00, "r": 0.5, "risk": 0.70, "label": "Bekasi"},
        {"lat": -6.97, "lon": 110.42, "r": 0.4, "risk": 0.75, "label": "Semarang"},
        {"lat": -7.25, "lon": 112.75, "r": 0.4, "risk": 0.65, "label": "Surabaya"},
        {"lat": -3.32, "lon": 114.59, "r": 0.5, "risk": 0.85, "label": "Banjarmasin"},
        {"lat": -2.99, "lon": 104.76, "r": 0.4, "risk": 0.72, "label": "Palembang"},
        {"lat":  3.59, "lon":  98.67, "r": 0.4, "risk": 0.68, "label": "Medan"},
    ]
    safezones = [
        {"lat": -7.43, "lon": 109.23, "r": 0.8, "risk": 0.10, "label": "Pegunungan Banyumas"},
        {"lat": -6.91, "lon": 107.61, "r": 0.5, "risk": 0.30, "label": "Bandung"},
        {"lat": -7.80, "lon": 110.36, "r": 0.4, "risk": 0.25, "label": "Yogyakarta"},
        {"lat": -7.97, "lon": 112.63, "r": 0.4, "risk": 0.20, "label": "Malang"},
        {"lat": -8.34, "lon": 115.09, "r": 0.6, "risk": 0.18, "label": "Bali Tengah"},
    ]
    best_hot  = min(hotspots,  key=lambda z: math.hypot(lat-z["lat"], lon-z["lon"]))
    best_safe = min(safezones, key=lambda z: math.hypot(lat-z["lat"], lon-z["lon"]))
    d_hot  = math.hypot(lat - best_hot["lat"],  lon - best_hot["lon"])
    d_safe = math.hypot(lat - best_safe["lat"], lon - best_safe["lon"])
    if d_safe < best_safe["r"] and (d_hot >= best_hot["r"] or d_safe < d_hot):
        return {"base_risk": best_safe["risk"], "label": best_safe["label"], "type": "safe"}
    if d_hot < best_hot["r"]:
        return {"base_risk": best_hot["risk"], "label": best_hot["label"], "type": "hotspot"}
    is_low = (-3 < lat < 4 and 108 < lon < 117) or (-5 < lat < 6 and 95 < lon < 108)
    return {"base_risk": 0.55 if is_low else 0.30, "label": "Dataran Rendah" if is_low else "Wilayah Umum", "type": "generic"}


def simulate_weather(lat: float, lon: float) -> dict:
    geo = geo_risk_profile(lat, lon)
    br = geo["base_risk"]
    is_rainy = datetime.utcnow().month in (10, 11, 12, 1, 2, 3, 4)
    rain_ceiling = 38 if is_rainy else 18
    rain = max(0.0, min(rain_ceiling, br * rain_ceiling + (random.random() - 0.3) * rain_ceiling * 0.6))
    return {
        "rainfall_intensity": round(rain, 1),
        "rainfall_duration":  round(random.random() * 5 * br + random.random() * 1.5, 1),
        "humidity":           round(55 + br * 30 + random.random() * 15, 1),
        "pressure":           round(1016 - br * 10 - random.random() * 4, 1),
        "temperature":        round(23 + random.random() * 10, 1),
        "wind_speed":         round(random.random() * 10, 1),
        "forecast_rain_24h":  round(max(0, min(60, br * 50 + (random.random() - 0.3) * 20)), 1),
    }


async def fetch_real_weather(lat: float, lon: float) -> dict:
    if not OPENWEATHER_API_KEY:
        return simulate_weather(lat, lon)
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
            )
            r.raise_for_status()
            d = r.json()
            rain_1h = d.get("rain", {}).get("1h", 0)
            fr = await client.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric", "cnt": 8}
            )
            fr.raise_for_status()
            fd = fr.json()
            forecast_rain = sum(i.get("rain", {}).get("3h", 0) for i in fd.get("list", []))
            return {
                "rainfall_intensity": round(rain_1h, 1),
                "rainfall_duration": 1.0,
                "humidity": d["main"]["humidity"],
                "pressure": round(d["main"]["pressure"], 1),
                "temperature": round(d["main"]["temp"], 1),
                "wind_speed": round(d["wind"]["speed"], 1),
                "forecast_rain_24h": round(forecast_rain, 1),
            }
    except Exception:
        return simulate_weather(lat, lon)


def get_nearest_river(lat: float, lon: float) -> str:
    rivers = [
        ((-6.14, 106.81), "Sungai Ciliwung"),
        ((-6.18, 106.75), "Sungai Pesanggrahan"),
        ((-6.20, 106.89), "Kali Sunter"),
        ((-6.16, 106.85), "Kali Malang"),
        ((-6.12, 106.78), "Kali Angke"),
        ((-6.22, 106.83), "Kali Krukut"),
    ]
    nearest, min_d = "Sungai Ciliwung", float("inf")
    for (rlat, rlon), name in rivers:
        d = math.sqrt((lat - rlat) ** 2 + (lon - rlon) ** 2)
        if d < min_d:
            min_d, nearest = d, name
    return nearest


def get_flood_zone(lat: float, lon: float) -> str:
    zones = ["Zona Merah (Rawan Tinggi)", "Zona Kuning (Rawan Sedang)", "Zona Hijau (Relatif Aman)"]
    if -6.20 <= lat <= -6.10 and 106.75 <= lon <= 106.95:
        return random.choices(zones, weights=[0.35, 0.40, 0.25])[0]
    return "Zona Kuning (Rawan Sedang)"


async def analyze_with_groq(weather: dict, location: LocationRequest, river: str, zone: str) -> dict:
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        rain = weather["rainfall_intensity"]
        dur  = weather["rainfall_duration"]
        pres = weather["pressure"]
        fore = weather["forecast_rain_24h"]
        score = min(100, int(rain * 1.2 + dur * 3 + (1015 - pres) * 0.8 + fore * 0.4))
        level, color = get_risk_level(score)
        analyses = {
            "LOW":      "Kondisi cuaca relatif aman. Tidak ada indikasi signifikan potensi banjir dalam waktu dekat.",
            "MEDIUM":   "Hujan cukup lebat terdeteksi. Pemantauan cuaca disarankan dan siapkan perlengkapan darurat.",
            "HIGH":     "Curah hujan tinggi dengan tekanan udara rendah. Potensi banjir signifikan dalam 3–6 jam ke depan.",
            "CRITICAL": "BAHAYA: Hujan ekstrem terdeteksi! Potensi banjir dalam 1–3 jam. Segera lakukan tindakan evakuasi.",
        }
        recs = {
            "LOW":      ["Pantau cuaca setiap 3 jam", "Pastikan saluran drainase bersih", "Simpan nomor kontak darurat"],
            "MEDIUM":   ["Siapkan tas darurat", "Amankan dokumen penting ke lantai atas", "Pantau ketinggian air sungai terdekat"],
            "HIGH":     ["Pindahkan barang berharga ke tempat tinggi", "Hubungi BPBD setempat", "Siapkan jalur evakuasi", "Isi cadangan air bersih"],
            "CRITICAL": ["SEGERA EVAKUASI ke tempat aman!", "Hubungi 119 atau BPBD sekarang", "Jangan melewati jalan tergenang", "Bawa dokumen & obat-obatan penting"],
        }
        return {"risk_score": score, "risk_level": level, "risk_color": color,
                "analysis": analyses[level], "recommendations": recs[level], "source": "rule-based"}

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""Kamu adalah sistem AI analisis risiko banjir FloodGuard.

Data cuaca:
- Intensitas hujan: {weather['rainfall_intensity']} mm/jam
- Durasi hujan: {weather['rainfall_duration']} jam  
- Kelembapan: {weather['humidity']}%
- Tekanan udara: {weather['pressure']} hPa
- Suhu: {weather['temperature']}°C
- Kecepatan angin: {weather['wind_speed']} m/s
- Prakiraan hujan 24 jam: {weather['forecast_rain_24h']} mm

Lokasi: {location.city or f'{location.lat},{location.lon}'}
Sungai terdekat: {river}
Zona historis: {zone}

Balas HANYA dengan JSON (tanpa teks lain, tanpa markdown):
{{"risk_score":<0-100>,"analysis":"<2-3 kalimat bahasa Indonesia>","recommendations":["<rec1>","<rec2>","<rec3>"]}}"""

    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        score = max(0, min(100, int(data.get("risk_score", 0))))
        level, color = get_risk_level(score)
        return {"risk_score": score, "risk_level": level, "risk_color": color,
                "analysis": data.get("analysis", ""), "recommendations": data.get("recommendations", []),
                "source": "groq-ai"}
    except Exception:
        score = min(100, int(weather["rainfall_intensity"] * 1.5 + weather["forecast_rain_24h"] * 0.5))
        level, color = get_risk_level(score)
        return {"risk_score": score, "risk_level": level, "risk_color": color,
                "analysis": "Analisis cadangan aktif.", "recommendations": ["Pantau cuaca secara berkala"],
                "source": "fallback"}


@app.get("/")
def root():
    return {"status": "FloodGuard AI API v1.0", "docs": "/docs"}


@app.post("/api/analyze", response_model=RiskResponse)
async def analyze(req: LocationRequest):
    weather   = await fetch_real_weather(req.lat, req.lon)
    river     = get_nearest_river(req.lat, req.lon)
    zone      = get_flood_zone(req.lat, req.lon)
    ai        = await analyze_with_groq(weather, req, river, zone)
    ts        = datetime.utcnow().isoformat()

    record = {
        "timestamp": ts, "lat": req.lat, "lon": req.lon,
        "city": req.city, "risk_score": ai["risk_score"],
        "risk_level": ai["risk_level"], "rainfall": weather["rainfall_intensity"],
        "humidity": weather["humidity"],
    }
    warnings_history.append(record)
    if len(warnings_history) > 100:
        warnings_history.pop(0)

    return RiskResponse(
        risk_score=ai["risk_score"], risk_level=ai["risk_level"],
        risk_color=ai["risk_color"], analysis=ai["analysis"],
        recommendations=ai["recommendations"], nearest_river=river,
        flood_zone=zone, weather=weather, timestamp=ts, source=ai["source"],
    )


@app.get("/api/history")
def history():
    return {"warnings": list(reversed(warnings_history[-30:])), "total": len(warnings_history)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq": bool(GROQ_API_KEY),
        "weather_api": "openweathermap" if OPENWEATHER_API_KEY else "simulation",
        "records": len(warnings_history),
        "ts": datetime.utcnow().isoformat(),
    }
