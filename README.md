# MushroomOS 🍄

An autonomous mushroom cultivation system running on a Raspberry Pi Zero 2W — built from scratch with computer vision, AI natural language control, and a live web dashboard.

> **Live dashboard:** [arjunmurugesan369-create.github.io/MushroomOS](https://arjunmurugesan369-create.github.io/MushroomOS)
> **Live API:** [mushroomos.uk/api/status](https://mushroomos.uk/api/status)

---

## What it does

MushroomOS autonomously manages a grey oyster mushroom (*Pleurotus ostreatus*) grow chamber — monitoring environmental conditions, controlling devices, detecting growth stages via computer vision, and allowing full natural language control through Telegram.

- Reads temperature and humidity every 5 minutes via SHT31 sensor (I2C, 0x45)
- Controls heater and fogger/fan via Tuya smart plugs using tinytuya
- Two control modes: **auto** (humidity/temp threshold) and **cycle** (timed ON/OFF for fruiting)
- Captures hourly chamber images and saves to SQLite with full telemetry
- Runs YOLOv8 (ONNX, 94% accuracy) every 2 hours to detect growth stage
- Sends Telegram alerts when pins appear or mushrooms are ready to harvest
- Answers natural language questions about historical data via Gemini AI + SQLite
- Serves a live REST API via Flask behind a permanent Cloudflare tunnel
- Hosts a live dashboard on GitHub Pages pulling real sensor data

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Raspberry Pi Zero 2W                   │
│                                                     │
│  mushroomos.py                                      │
│  ├── SHT31 sensor (I2C)                             │
│  ├── Tuya smart plugs (tinytuya)                    │
│  ├── Auto-control loop (5 min cycle)                │
│  ├── Cycle control loop (timed ON/OFF)              │
│  ├── Hourly data collection + camera capture        │
│  ├── YOLOv8 ONNX inference (every 2 snapshots)      │
│  ├── Telegram bot (python-telegram-bot)             │
│  │   └── Gemini AI router (database/action/chat)   │
│  └── Flask REST API (port 5000)                     │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │ Cloudflare Tunnel
                   ▼
            mushroomos.uk (HTTPS)
                   │
                   ├── /api/status
                   ├── /api/relays
                   ├── /api/history
                   ├── /api/snapshot
                   ├── /api/relay/<device>
                   ├── /api/targets
                   ├── /api/mode
                   └── /api/stage

GitHub Pages → index.html (dashboard) → fetches from mushroomos.uk
```

---

## Hardware

| Component | Details |
|---|---|
| Compute | Raspberry Pi Zero 2W (512MB RAM) |
| Sensor | SHT31-D — temperature + humidity (I2C, address 0x45) |
| Devices | Tuya smart plugs × 2 (heater + fogger/fan) |
| Camera | IP camera via OpenCV (MJPEG stream) |
| Storage | microSD 32GB |

**Wiring (SHT31):**
- Pin 1 (3.3V) → VCC
- Pin 9 (GND) → GND
- Pin 3 (SDA) → SDA
- Pin 5 (SCL) → SCL

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Bot framework | python-telegram-bot (async) |
| AI | Gemini 2.5 Flash + Gemini 2.5 Flash Lite |
| Computer vision | YOLOv8 via ONNX Runtime (no PyTorch on Pi) |
| Device control | tinytuya |
| Sensor | adafruit-circuitpython-sht31d |
| API | Flask + flask-cors |
| Database | SQLite |
| Tunnel | Cloudflare Tunnel (permanent HTTPS) |
| Dashboard | Vanilla HTML/CSS/JS on GitHub Pages |
| Process manager | systemd |

---

## AI Architecture

Three-way intent router classifies every Telegram message:

```
User message
     │
     ▼
Route intent (gemini-2.5-flash-lite, fast)
     │
     ├── database → Generate SQL → Run query → Answer with Gemini Flash
     ├── action   → Generate JSON action → Execute on hardware
     └── chat     → Gemini 2.5 Flash with conversation history + live sensor context
```

**Example interactions:**
- `"it's too humid"` → adjusts humidity target, saves to DB
- `"turn the fan off in 20 minutes"` → scheduled async task
- `"what's the avg temp last 3 hours?"` → SQL query + natural language answer
- `"is 98% humidity ok for fruiting?"` → expert mycology advice with live context
- `"take a pic"` → captures image, runs YOLO, returns detection in caption
- `"switch to cycle mode 15 on 15 off"` → sets timed fogger cycle

---

## YOLO Model

- Architecture: YOLOv8
- Format: ONNX (exported for lightweight inference — no PyTorch required)
- Classes: `empty`, `young`, `ready`, `old`
- Accuracy: 94%
- Inference: ~30s on RPi Zero 2W via onnxruntime CPU
- Trigger: every 2nd hourly snapshot + every manual photo

**Stage mapping:**
| Class | Stage | Action |
|---|---|---|
| empty | Colonisation | No change |
| young | Pinning | Alert + update stage |
| ready | Fruiting | Harvest alert |
| old | Post-harvest | Urgent alert |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | Live sensors, targets, stage, mode, uptime |
| GET | `/api/relays` | Heater and fogger states + power draw |
| GET | `/api/history?hours=24` | Historical snapshots from SQLite |
| GET | `/api/snapshot` | Most recent chamber image (JPEG) |
| POST | `/api/relay/<device>` | Toggle heater or fogger |
| POST | `/api/targets` | Set temp/humidity targets |
| POST | `/api/mode` | Switch auto/cycle mode, set cycle timing |
| POST | `/api/stage` | Update growth stage |

---

## Setup

**Install dependencies:**
```bash
pip install python-telegram-bot tinytuya google-generativeai \
            flask flask-cors opencv-python \
            adafruit-circuitpython-sht31d \
            onnxruntime numpy pillow python-dotenv
```

**Environment variables — create `.env`:**
```
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
HEATER_ID=your_tuya_device_id
HEATER_IP=192.168.x.x
HEATER_KEY=your_tuya_local_key
FOGGER_ID=your_tuya_device_id
FOGGER_IP=192.168.x.x
FOGGER_KEY=your_tuya_local_key
```

**Run:**
```bash
python3 mushroomos.py
```

**Run 24/7 with systemd:**
```bash
sudo systemctl enable mushroomos
sudo systemctl start mushroomos
```

**Cloudflare tunnel (permanent):**
```bash
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

---

## Database Schema

```sql
CREATE TABLE hourly_snapshots (
    snapshot_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT,
    grow_id           TEXT DEFAULT 'grow_001',
    hours_since_start INTEGER,
    temperature_c     REAL,
    humidity_pct      REAL,
    heater_on         BOOLEAN,
    fogger_fan_on     BOOLEAN,
    heater_power_w    REAL,
    fogger_power_w    REAL,
    image_filename    TEXT,
    yolo_class        TEXT,
    yolo_confidence   REAL
);

CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
```

---

## Roadmap

- [ ] GPIO relay for independent fan control (currently fan + fogger share one plug)
- [ ] Pre-capture ventilation sequence (fan on → wait → photo → fan off)
- [ ] YOLO confidence trend tracking across flushes
- [ ] Multi-species parameter profiles
- [ ] Timelapse video generation from hourly images
- [ ] Mobile-optimised dashboard

---

## Author

**Arjun Murugesan** — Fermentation Scientist & Bioprocess Engineer

3 years upstream R&D experience (E. coli, Pichia pastoris, Bacillus subtilis, 1–50L bioreactor scale) at Anthem Biosciences. MSc Project Management, Anglia Ruskin University 2024. Self-taught hardware/software development: Raspberry Pi, Arduino, Flutter, computer vision, 3D printing.

[GitHub](https://github.com/arjunmurugesan369-create) · [mushroomos.uk](https://mushroomos.uk)
