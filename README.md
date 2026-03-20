# MushroomOS 🍄

An autonomous mushroom cultivation system running on a Raspberry Pi Zero 2W, controlled via Telegram with Gemini AI natural language interface.

## Features

- Live environmental monitoring via SHT31 sensor (temperature + humidity)
- Automated bang-bang control of heater and fogger/fan via Tuya smart plugs
- Hourly data logging to SQLite with timelapse image capture
- Telegram bot for remote control from anywhere
- Gemini AI (gemini-2.5-flash-lite) for plain English commands
- Scheduled actions — "turn the fan off in 30 minutes"
- Natural language image capture — "take a pic"

## Hardware

| Component | Details |
|---|---|
| Compute | Raspberry Pi Zero 2W |
| Sensor | SHT31-D (I2C, address 0x45) |
| Devices | Tuya smart plugs x2 |
| Camera | IP camera via OpenCV |

## Stack

- Python 3 + asyncio
- python-telegram-bot
- tinytuya
- google-generativeai
- OpenCV
- SQLite

## Setup
```bash
pip install python-telegram-bot tinytuya google-generativeai opencv-python adafruit-circuitpython-sht31d python-dotenv
```

Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
nano .env
```

Run:
```bash
python3 mushroomos.py
```

## Environment Variables

Create a `.env` file with:
```
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
HEATER_ID=your_tuya_device_id
HEATER_IP=your_tuya_device_ip
HEATER_KEY=your_tuya_local_key
FOGGER_ID=your_tuya_device_id
FOGGER_IP=your_tuya_device_ip
FOGGER_KEY=your_tuya_local_key
```

## Running 24/7 with systemd
```bash
sudo systemctl enable mushroomos
sudo systemctl start mushroomos
```

## Telegram Commands

| Command | Action |
|---|---|
| `/status` | Live sensor readings and device states |
| `/pic` | Capture chamber image |
| `/data` | Last 5 hourly snapshots |
| `/export` | Download full dataset as CSV |
| `"it's too humid"` | AI adjusts humidity target |
| `"turn the fan off in 20 minutes"` | Scheduled action |
| `"take a pic"` | Natural language image capture |

## Author

Arjun Murugesan — Fermentation Scientist & Bioprocess Engineer
