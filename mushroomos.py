import logging
import asyncio
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import board
import adafruit_sht31d
import tinytuya
import cv2
import time
from datetime import datetime
import sqlite3
import csv
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
import google.generativeai as genai
from collections import deque

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ─── GEMINI CONFIG ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
genai.configure(api_key=GEMINI_API_KEY)

gemini_lite = genai.GenerativeModel("gemini-2.5-flash-lite")
gemini_chat = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="""You are an expert mushroom cultivation assistant and the AI brain of MushroomOS —
an autonomous grow chamber system built on a Raspberry Pi Zero 2W.

You have deep knowledge of mycology, mushroom biology, environmental control, and cultivation best practices.
You are helping grow grey oyster mushrooms (Pleurotus ostreatus).

Your personality:
- Warm, direct, and genuinely helpful — like a knowledgeable friend, not a corporate chatbot
- Proactive — if you notice something concerning in the sensor data, mention it
- Concise but complete — don't pad responses, but don't truncate important info either
- Use Telegram markdown: *bold* for emphasis, bullet points for lists, `code` for numbers/values

You have access to live sensor readings and the current growth stage provided in each message.
The grow chamber has a YOLOv8 computer vision system (running via ONNX) that detects:
- empty  → colonisation (bare substrate, no mushrooms)
- young  → pinning (early pins forming)
- ready  → fruiting (mature, harvest soon)
- old    → post-harvest (overripe, missed window)

When giving advice always factor in the current conditions and growth stage."""
)

# ─── YOLO CONFIG ───────────────────────────────────────────────
YOLO_MODEL_PATH = os.path.expanduser("~/best.onnx")
YOLO_CONFIDENCE = 0.15
YOLO_SESSION    = None

# Class names — must match training order
# Check your model's actual order with: session.get_outputs()[0].shape
YOLO_CLASSES = ["empty", "old", "ready", "young"]

YOLO_STAGE_MAP = {
    "empty": "colonisation",
    "young": "pinning",
    "ready": "fruiting",
    "old":   "post-harvest",
}

STAGE_ORDER = ["colonisation", "pinning", "fruiting", "post-harvest"]

def load_yolo_model():
    global YOLO_SESSION
    if YOLO_SESSION is None:
        if os.path.exists(YOLO_MODEL_PATH):
            print(f"🔍 Loading ONNX model from {YOLO_MODEL_PATH}...")
            YOLO_SESSION = ort.InferenceSession(
                YOLO_MODEL_PATH,
                providers=['CPUExecutionProvider']
            )
            print("✅ ONNX model loaded")
        else:
            print(f"⚠️  ONNX model not found at {YOLO_MODEL_PATH} — detection disabled")
    return YOLO_SESSION

def preprocess_image(image_path: str, imgsz: int = 640):
    """Load and preprocess image for YOLOv8 ONNX inference."""
    img       = Image.open(image_path).convert('RGB')
    img       = img.resize((imgsz, imgsz))
    arr       = np.array(img, dtype=np.float32) / 255.0
    arr       = arr.transpose(2, 0, 1)        # HWC → CHW
    arr       = np.expand_dims(arr, axis=0)   # add batch dim
    return arr

def postprocess(outputs, conf_threshold: float = 0.5):
    """
    Parse YOLOv8 ONNX output.
    Output shape: [1, num_classes+4, num_anchors]
    Returns (best_class_name, best_confidence) or (None, None).
    """
    output       = outputs[0][0]         # shape: (8, 8400) for 4 classes
    class_scores = output[4:, :]         # rows 4+ are class probabilities

    best_conf  = 0.0
    best_class = None

    for anchor_idx in range(class_scores.shape[1]):
        scores    = class_scores[:, anchor_idx]
        max_score = float(np.max(scores))
        max_idx   = int(np.argmax(scores))

        if max_score > conf_threshold and max_score > best_conf:
            best_conf  = max_score
            best_class = YOLO_CLASSES[max_idx] if max_idx < len(YOLO_CLASSES) else str(max_idx)

    return best_class, round(best_conf, 3)

async def run_yolo(image_path: str) -> tuple:
    """Run ONNX inference asynchronously. Returns (class, confidence) or (None, None)."""
    session = load_yolo_model()
    if session is None:
        return None, None

    loop = asyncio.get_event_loop()

    def _infer():
        img_array  = preprocess_image(image_path)
        input_name = session.get_inputs()[0].name
        outputs    = session.run(None, {input_name: img_array})
        return postprocess(outputs, YOLO_CONFIDENCE)

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _infer),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        print("⚠️  YOLO inference timed out")
        return None, None
    except Exception as e:
        print(f"⚠️  YOLO error: {e}")
        return None, None

async def handle_detection(image_path: str, bot=None) -> tuple:
    """Run YOLO, update stage if needed, send Telegram alerts."""
    global GROWTH_STAGE

    yolo_class, yolo_conf = await run_yolo(image_path)

    if yolo_class is None:
        print("🔍 YOLO: no detection above threshold")
        return None, None

    print(f"🔍 YOLO: {yolo_class} ({yolo_conf:.1%})")

    detected_stage = YOLO_STAGE_MAP.get(yolo_class, "colonisation")
    current_idx    = STAGE_ORDER.index(GROWTH_STAGE)  if GROWTH_STAGE  in STAGE_ORDER else 0
    detected_idx   = STAGE_ORDER.index(detected_stage) if detected_stage in STAGE_ORDER else 0
    stage_changed  = detected_idx > current_idx

    alert_messages = {
        "young": (
            f"🍄 *Pins detected!*\n\n"
            f"Confidence: *{yolo_conf:.1%}*\n"
            f"Stage updated: *Colonisation → Pinning*\n\n"
            f"Recommended now:\n"
            f"• Increase fresh air exchange\n"
            f"• Raise humidity to 90-95%\n"
            f"• Ensure 12hr light cycle\n"
            f"• Drop temp to 18-22°C for grey oyster"
        ),
        "ready": (
            f"🎉 *Mushrooms ready to harvest!*\n\n"
            f"Confidence: *{yolo_conf:.1%}*\n"
            f"Stage updated: *Pinning → Fruiting*\n\n"
            f"Harvest when caps are fully open but *before* edges curl.\n"
            f"Twist and pull — don't cut."
        ),
        "old": (
            f"⚠️ *Mushrooms past harvest window!*\n\n"
            f"Confidence: *{yolo_conf:.1%}*\n"
            f"Caps are over-mature and may sporulate.\n"
            f"Harvest immediately and clean the block thoroughly."
        ),
        "empty": None
    }

    if stage_changed:
        GROWTH_STAGE = detected_stage
        save_setting("growth_stage", GROWTH_STAGE)
        print(f"🌱 Stage updated to: {GROWTH_STAGE}")

    if bot and ALERT_CHAT_IDS:
        alert_msg = alert_messages.get(yolo_class)
        if alert_msg and (stage_changed or yolo_class == "old"):
            for chat_id in ALERT_CHAT_IDS:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=alert_msg,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    print(f"⚠️  Alert error: {e}")

    return yolo_class, yolo_conf

async def capture_and_detect(bot=None) -> tuple:
    """Capture image + run YOLO. Returns (filename, yolo_class, yolo_conf)."""
    loop = asyncio.get_event_loop()

    def _capture():
        cap = cv2.VideoCapture(CAMERA_URL)
        ret, frame = cap.read()
        cap.release()
        if ret:
            fn = f'temp_{int(time.time())}.jpg'
            cv2.imwrite(fn, frame)
            return fn
        return None

    filename = await loop.run_in_executor(None, _capture)
    if not filename:
        return None, None, None

    yolo_class, yolo_conf = await handle_detection(filename, bot=bot)
    return filename, yolo_class, yolo_conf

# ─── HARDWARE ──────────────────────────────────────────────────
i2c    = board.I2C()
sensor = adafruit_sht31d.SHT31D(i2c, address=0x45)

heater = tinytuya.OutletDevice(
    dev_id=os.getenv("HEATER_ID",  "bfb03563ee57c75f5eizxe"),
    address=os.getenv("HEATER_IP", "192.168.1.60"),
    local_key=os.getenv("HEATER_KEY", "~|:cS^1q?i/1ag7W"),
    version=3.5
)
heater.set_socketPersistent(False)

fogger_fan = tinytuya.OutletDevice(
    dev_id=os.getenv("FOGGER_ID",  "bfa9685b73fcad1bc64sdn"),
    address=os.getenv("FOGGER_IP", "192.168.1.249"),
    local_key=os.getenv("FOGGER_KEY", "zCMzT{]h==&V=e?A"),
    version=3.5
)
fogger_fan.set_socketPersistent(False)

CAMERA_URL = 'http://192.168.1.33:8080/video'

# ─── DATABASE ──────────────────────────────────────────────────
DB_PATH = os.path.expanduser('~/mushroom_data.db')
conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor  = conn.cursor()
os.makedirs('images', exist_ok=True)

DB_SCHEMA = """
Table: hourly_snapshots
Columns:
  snapshot_id       INTEGER PRIMARY KEY
  timestamp         TEXT    (format: 'YYYY-MM-DD HH:MM:SS', stored in local time)
  grow_id           TEXT
  hours_since_start INTEGER
  temperature_c     REAL
  humidity_pct      REAL
  heater_on         BOOLEAN (0=off, 1=on)
  fogger_fan_on     BOOLEAN (0=off, 1=on)
  heater_power_w    REAL
  fogger_power_w    REAL
  image_filename    TEXT
  yolo_class        TEXT    (empty/young/ready/old, NULL if no detection)
  yolo_confidence   REAL    (0-1, NULL if no detection)

Table: settings
Columns:
  key   TEXT PRIMARY KEY
  value TEXT
  (stores: target_temp, target_humidity, growth_stage)

For time queries use: datetime('now', '-X hours') — timestamps are local time and datetime('now') matches.
"""

def init_db():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    for col in ["yolo_class TEXT", "yolo_confidence REAL"]:
        try:
            cursor.execute(f"ALTER TABLE hourly_snapshots ADD COLUMN {col}")
        except Exception:
            pass
    cursor.execute("INSERT OR IGNORE INTO settings VALUES ('target_temp',     '23.0')")
    cursor.execute("INSERT OR IGNORE INTO settings VALUES ('target_humidity', '70.0')")
    cursor.execute("INSERT OR IGNORE INTO settings VALUES ('growth_stage',    'colonisation')")
    conn.commit()

def load_settings():
    cursor.execute("SELECT key, value FROM settings")
    return {row[0]: row[1] for row in cursor.fetchall()}

def save_setting(key: str, value: str):
    cursor.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))
    conn.commit()

# ─── STATE ─────────────────────────────────────────────────────
settings        = {}
hour_counter    = 0
start_time      = time.time()
TARGET_TEMP     = 23.0
TARGET_HUMIDITY = 70.0
GROWTH_STAGE    = "colonisation"
AUTO_CONTROL    = True
ALERT_CHAT_IDS: set = set()

conversation_history = {}
MAX_HISTORY = 20

def get_history(chat_id: int):
    if chat_id not in conversation_history:
        conversation_history[chat_id] = deque(maxlen=MAX_HISTORY)
    return conversation_history[chat_id]

# ─── ASYNC TUYA HELPERS ────────────────────────────────────────

async def tuya_status(device):
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, device.status), timeout=5.0
        )
    except Exception:
        return None

async def tuya_on(device):
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, device.turn_on), timeout=5.0
        )
        return True
    except Exception:
        return False

async def tuya_off(device):
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, device.turn_off), timeout=5.0
        )
        return True
    except Exception:
        return False

def read_sensor():
    try:
        return round(sensor.temperature, 1), round(sensor.relative_humidity, 1)
    except Exception:
        return None, None

# ─── INTENT ROUTER ─────────────────────────────────────────────

async def route_intent(message: str) -> str:
    prompt = f"""Classify this message into exactly one category. Reply with only the single word.

Categories:
- database: questions about historical data, past readings, averages, trends, "last X hours", "how has", "what was", "show me data", "average", "history", "snapshots", "when did"
- action: device control, turn on/off, scheduling, setting targets, taking photos, changing parameters, "set", "turn", "take a pic", "schedule", "change target", "set stage"
- chat: mushroom advice, cultivation questions, biology, troubleshooting, general conversation, "why", "how does", "what should I", "is this normal", "explain"

Message: {message}

Reply with only one word: database, action, or chat"""

    loop     = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: gemini_lite.generate_content(prompt))
    result   = response.text.strip().lower()
    return result if result in ["database", "action", "chat"] else "chat"

# ─── DATABASE Q&A ──────────────────────────────────────────────

async def ask_database(user_message: str) -> str:
    loop = asyncio.get_event_loop()

    sql_prompt = f"""You are a SQLite expert. Generate a single safe SELECT query to answer the user's question.

DATABASE SCHEMA:
{DB_SCHEMA}

RULES:
- Only SELECT statements — never INSERT, UPDATE, DELETE, DROP
- For "last X hours" use: datetime('now', '-X hours')
- Always LIMIT to 100 rows max
- Return ONLY the raw SQL, no markdown, no explanation

User question: {user_message}

SQL:"""

    sql_response = await loop.run_in_executor(None, lambda: gemini_lite.generate_content(sql_prompt))
    sql = sql_response.text.strip()

    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    sql = sql.strip()

    if not sql.upper().startswith("SELECT"):
        return "Sorry, I could only generate a safe read-only query. Try rephrasing."

    try:
        cursor.execute(sql)
        rows    = cursor.fetchall()
        columns = [d[0] for d in cursor.description]
    except Exception as e:
        return f"❌ Query error: `{str(e)}`\n\nSQL:\n`{sql}`"

    if not rows:
        return f"📭 No data found for that period.\n\nSQL used:\n`{sql}`"

    results_text  = "Columns: " + ", ".join(columns) + "\n"
    results_text += "\n".join([str(dict(zip(columns, row))) for row in rows[:50]])

    answer_prompt = f"""The user asked: "{user_message}"

Query results:
{results_text}

Give a clear conversational answer. Use *bold* for key numbers.
Mention any concerning trends. Use Telegram markdown."""

    answer = await loop.run_in_executor(None, lambda: gemini_chat.generate_content(answer_prompt))
    return answer.text.strip()

# ─── ACTION HANDLER ────────────────────────────────────────────

async def ask_action(user_message: str) -> dict:
    temp, humidity = read_sensor()
    heater_s       = await tuya_status(heater)
    fogger_s       = await tuya_status(fogger_fan)
    heater_state   = heater_s['dps']['1'] if heater_s else "unknown"
    fogger_state   = fogger_s['dps']['1'] if fogger_s else "unknown"

    prompt = f"""You control a grey oyster mushroom grow chamber.

LIVE READINGS:
  Temperature  : {f'{temp}°C' if temp else 'unavailable'}  (target: {TARGET_TEMP}°C)
  Humidity     : {f'{humidity}%' if humidity else 'unavailable'}  (target: {TARGET_HUMIDITY}%)
  Growth stage : {GROWTH_STAGE}

DEVICE STATES:
  Heater     : {'ON' if heater_state is True else 'OFF' if heater_state is False else heater_state}
  Fogger/Fan : {'ON' if fogger_state is True else 'OFF' if fogger_state is False else fogger_state}

AUTO-CONTROL: {'ON' if AUTO_CONTROL else 'OFF'}

Respond with ONLY a single JSON object — no markdown, no code fences, no extra text.

Valid actions:
{{"action":"toggle_heater","state":true,"reply":"Heater turned on ✅"}}
{{"action":"toggle_fogger","state":false,"reply":"Fogger turned off ✅"}}
{{"action":"set_temp","value":22.0,"reply":"Target temperature set to 22°C and saved ✅"}}
{{"action":"set_humidity","value":75.0,"reply":"Target humidity set to 75% and saved ✅"}}
{{"action":"set_stage","stage":"pinning","reply":"Growth stage updated to pinning ✅"}}
{{"action":"set_auto","state":true,"reply":"Auto-control enabled ✅"}}
{{"action":"schedule","device":"fogger","state":false,"delay_minutes":5,"reply":"Fogger will turn off in 5 minutes ⏱️"}}
{{"action":"capture","reply":"Capturing and analysing chamber image 📸"}}
{{"action":"none","reply":"Your answer here"}}

Rules:
- Convert hours to minutes for scheduling
- Temp 15-28°C, humidity 60-98%
- Valid stages: colonisation, pinning, fruiting, post-harvest
- One sentence reply max

User message: {user_message}"""

    loop     = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: gemini_lite.generate_content(prompt))
    raw      = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

# ─── CHAT HANDLER ──────────────────────────────────────────────

async def ask_chat(user_message: str, chat_id: int) -> str:
    temp, humidity = read_sensor()
    loop           = asyncio.get_event_loop()
    history        = get_history(chat_id)

    context = f"""[Chamber context — {datetime.now().strftime('%H:%M, %d %b %Y')}]
Temperature  : {f'{temp}°C' if temp else 'unavailable'} (target: {TARGET_TEMP}°C)
Humidity     : {f'{humidity}%' if humidity else 'unavailable'} (target: {TARGET_HUMIDITY}%)
Growth stage : {GROWTH_STAGE}
Auto-control : {'ON' if AUTO_CONTROL else 'OFF'}

{user_message}"""

    messages = list(history) + [{"role": "user", "parts": [context]}]
    response = await loop.run_in_executor(None, lambda: gemini_chat.generate_content(messages))
    reply    = response.text.strip()

    history.append({"role": "user",  "parts": [context]})
    history.append({"role": "model", "parts": [reply]})

    return reply

# ─── AUTO CONTROL LOOP ─────────────────────────────────────────

async def auto_control_loop():
    print("🤖 Auto-control started")
    while True:
        if AUTO_CONTROL:
            try:
                temp, humidity = read_sensor()
                if temp is None:
                    await asyncio.sleep(60)
                    continue

                if temp < TARGET_TEMP - 1:
                    await tuya_on(heater)
                    print(f"🔥 Heater ON ({temp}°C < {TARGET_TEMP}°C)")
                elif temp > TARGET_TEMP + 1:
                    await tuya_off(heater)
                    print(f"❄️  Heater OFF ({temp}°C > {TARGET_TEMP}°C)")

                if humidity < TARGET_HUMIDITY - 3:
                    await tuya_on(fogger_fan)
                    print(f"💨 Fogger ON ({humidity}% < {TARGET_HUMIDITY}%)")
                elif humidity > TARGET_HUMIDITY + 3:
                    await tuya_off(fogger_fan)
                    print(f"💧 Fogger OFF ({humidity}% > {TARGET_HUMIDITY}%)")

            except Exception as e:
                print(f"❌ Auto-control error: {e}")

        await asyncio.sleep(300)

# ─── DATA COLLECTION ───────────────────────────────────────────

async def hourly_collection_task(app):
    global hour_counter
    while True:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n📊 Collecting snapshot #{hour_counter} at {timestamp}")
        try:
            temp, humidity = read_sensor()
            heater_s = await tuya_status(heater)
            fogger_s = await tuya_status(fogger_fan)

            heater_on    = heater_s['dps']['1']       if heater_s else False
            fogger_on    = fogger_s['dps']['1']       if fogger_s else False
            heater_power = heater_s['dps']['19'] / 10 if heater_s else 0
            fogger_power = fogger_s['dps']['19'] / 10 if fogger_s else 0

            image_filename = f"grow_001_h{hour_counter:03d}.jpg"
            loop = asyncio.get_event_loop()

            def capture():
                cap = cv2.VideoCapture(CAMERA_URL)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    path = f'images/{image_filename}'
                    cv2.imwrite(path, frame)
                    return path
                return None

            captured_path  = await loop.run_in_executor(None, capture)
            yolo_class, yolo_conf = None, None

            # Run YOLO every 2 hours (every 2nd snapshot)
            if captured_path and hour_counter % 2 == 0:
                yolo_class, yolo_conf = await handle_detection(
                    captured_path, bot=app.bot
                )

            cursor.execute('''
            INSERT INTO hourly_snapshots
            (timestamp, hours_since_start, temperature_c, humidity_pct,
             heater_on, fogger_fan_on, heater_power_w, fogger_power_w,
             image_filename, yolo_class, yolo_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, hour_counter, temp, humidity,
                  heater_on, fogger_on, heater_power, fogger_power,
                  image_filename, yolo_class, yolo_conf))
            conn.commit()

            yolo_str = f" | YOLO: {yolo_class} ({yolo_conf:.1%})" if yolo_class else ""
            print(f"✅ Snapshot #{hour_counter} saved{yolo_str}")
            hour_counter += 1

        except Exception as e:
            print(f"❌ Snapshot error: {e}")

        await asyncio.sleep(3600)

# ─── TELEGRAM COMMANDS ─────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ALERT_CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text(
        "🍄 *MushroomOS Active*\n\n"
        f"🌱 Stage: *{GROWTH_STAGE.capitalize()}*\n"
        f"🎯 Targets: *{TARGET_TEMP}°C* · *{TARGET_HUMIDITY}%*\n"
        f"🤖 Auto-control: {'✅ ON' if AUTO_CONTROL else '❌ OFF'}\n"
        f"🔍 YOLO: {'✅ ready' if os.path.exists(YOLO_MODEL_PATH) else '❌ model not found'}\n\n"
        "*Commands:*\n"
        "/status — live readings\n"
        "/pic — capture + analyse\n"
        "/stage — growth stage\n"
        "/data — recent snapshots\n"
        "/export — download CSV\n"
        "/clear — reset conversation\n"
        "/auto\\_on · /auto\\_off\n"
        "/heater\\_on · /heater\\_off\n"
        "/fogger\\_on · /fogger\\_off\n\n"
        "💬 *Just talk naturally:*\n"
        "_\"avg temp last 3 hours?\"_\n"
        "_\"set humidity to 85%\"_\n"
        "_\"is this ok for pinning?\"_\n"
        "_\"turn fan off in 20 mins\"_\n"
        "_\"take a pic\"_",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ALERT_CHAT_IDS.add(update.effective_chat.id)
    try:
        temp, humidity = read_sensor()
        heater_s  = await tuya_status(heater)
        fogger_s  = await tuya_status(fogger_fan)

        heater_on    = heater_s['dps']['1']       if heater_s else '?'
        fogger_on    = fogger_s['dps']['1']       if fogger_s else '?'
        heater_power = heater_s['dps']['19'] / 10 if heater_s else '?'
        fogger_power = fogger_s['dps']['19'] / 10 if fogger_s else '?'
        hours_running = int((time.time() - start_time) / 3600)

        temp_warn = " ⚠️" if temp     and (temp     < TARGET_TEMP     - 2 or temp     > TARGET_TEMP     + 2) else ""
        hum_warn  = " ⚠️" if humidity and (humidity < TARGET_HUMIDITY - 5 or humidity > TARGET_HUMIDITY + 5) else ""

        cursor.execute("""
            SELECT yolo_class, yolo_confidence, timestamp
            FROM hourly_snapshots
            WHERE yolo_class IS NOT NULL
            ORDER BY snapshot_id DESC LIMIT 1
        """)
        last = cursor.fetchone()
        yolo_line = f"\n🔍 Last detection: *{last[0]}* ({last[1]:.1%}) at `{last[2]}`" if last else ""

        await update.message.reply_text(
            f"🍄 *MushroomOS Status*\n\n"
            f"⏱️ Running: *{hours_running}h* · Snapshots: *{hour_counter}*\n"
            f"🌱 Stage: *{GROWTH_STAGE.capitalize()}*\n"
            f"🤖 Auto-control: {'✅ ON' if AUTO_CONTROL else '❌ OFF'}"
            f"{yolo_line}\n\n"
            f"🌡️ Temp: *{temp}°C*{temp_warn} (target: {TARGET_TEMP}°C)\n"
            f"💧 Humidity: *{humidity}%*{hum_warn} (target: {TARGET_HUMIDITY}%)\n\n"
            f"🔥 Heater: {'✅ ON' if heater_on is True else '❌ OFF'} ({heater_power}W)\n"
            f"💨 Fogger/Fan: {'✅ ON' if fogger_on is True else '❌ OFF'} ({fogger_power}W)",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def pic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ALERT_CHAT_IDS.add(update.effective_chat.id)
    try:
        await update.message.reply_text("📸 Capturing and analysing...")
        filename, yolo_class, yolo_conf = await capture_and_detect(bot=context.bot)
        if filename:
            caption = ""
            if yolo_class:
                stage   = YOLO_STAGE_MAP.get(yolo_class, "unknown")
                caption = f"🔍 *{yolo_class.capitalize()}* ({yolo_conf:.1%}) → *{stage}*"
            await update.message.reply_photo(
                photo=open(filename, 'rb'),
                caption=caption,
                parse_mode='Markdown'
            )
            os.remove(filename)
        else:
            await update.message.reply_text("❌ Failed to capture image")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def stage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cursor.execute("""
        SELECT yolo_class, yolo_confidence, timestamp
        FROM hourly_snapshots
        WHERE yolo_class IS NOT NULL
        ORDER BY snapshot_id DESC LIMIT 1
    """)
    last     = cursor.fetchone()
    last_str = f"\n🔍 Last YOLO: *{last[0]}* ({last[1]:.1%}) at `{last[2]}`" if last else ""
    await update.message.reply_text(
        f"🌱 Current stage: *{GROWTH_STAGE.capitalize()}*"
        f"{last_str}\n\n"
        f"YOLO runs every 2 hours automatically.\n"
        f"To set manually: _\"set stage to pinning\"_",
        parse_mode='Markdown'
    )

async def data_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cursor.execute("SELECT COUNT(*) FROM hourly_snapshots")
        count = cursor.fetchone()[0]
        if count == 0:
            await update.message.reply_text("📊 No data yet")
            return
        cursor.execute("""
        SELECT timestamp, temperature_c, humidity_pct, heater_on, fogger_fan_on, yolo_class
        FROM hourly_snapshots ORDER BY snapshot_id DESC LIMIT 5
        """)
        rows    = cursor.fetchall()
        message = f"📊 *Last {len(rows)} Snapshots* (Total: {count})\n\n"
        for row in rows:
            ts, temp, hum, heat, fog, yc = row
            yolo_str = f" 🔍{yc}" if yc else ""
            message += f"`{ts}` *{temp:.1f}°C* {hum:.1f}% H:{'✅' if heat else '❌'} F:{'✅' if fog else '❌'}{yolo_str}\n"
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def export_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("📊 Exporting...")
        cursor.execute("SELECT * FROM hourly_snapshots ORDER BY snapshot_id")
        rows = cursor.fetchall()
        if not rows:
            await update.message.reply_text("❌ No data yet")
            return
        filename = f'mushroom_data_{int(time.time())}.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID','Timestamp','Grow_ID','Hours','Temp_C','Humidity_%',
                             'Heater_ON','Fogger_ON','Heater_W','Fogger_W','Image',
                             'YOLO_Class','YOLO_Confidence'])
            writer.writerows(rows)
        await update.message.reply_document(
            document=open(filename, 'rb'),
            filename='mushroom_data.csv',
            caption=f"📊 {len(rows)} data points exported"
        )
        os.remove(filename)
    except Exception as e:
        await update.message.reply_text(f"❌ Export error: {str(e)}")

async def auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_CONTROL
    AUTO_CONTROL = True
    await update.message.reply_text("🤖 Auto-control *ENABLED*", parse_mode='Markdown')

async def auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_CONTROL
    AUTO_CONTROL = False
    await update.message.reply_text("⚠️ Auto-control *DISABLED*", parse_mode='Markdown')

async def heater_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_on(heater)
    await update.message.reply_text("🔥 Heater *ON*" if ok else "❌ Failed", parse_mode='Markdown')

async def heater_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_off(heater)
    await update.message.reply_text("❄️ Heater *OFF*" if ok else "❌ Failed", parse_mode='Markdown')

async def fogger_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_on(fogger_fan)
    await update.message.reply_text("💨 Fogger *ON*" if ok else "❌ Failed", parse_mode='Markdown')

async def fogger_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_off(fogger_fan)
    await update.message.reply_text("💧 Fogger *OFF*" if ok else "❌ Failed", parse_mode='Markdown')

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in conversation_history:
        conversation_history[chat_id].clear()
    await update.message.reply_text("🧹 Conversation history cleared.")

# ─── NATURAL LANGUAGE HANDLER ──────────────────────────────────

async def handle_natural(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TARGET_TEMP, TARGET_HUMIDITY, GROWTH_STAGE, AUTO_CONTROL

    chat_id = update.effective_chat.id
    ALERT_CHAT_IDS.add(chat_id)
    await context.bot.send_chat_action(chat_id, "typing")

    try:
        intent = await route_intent(update.message.text)

        if intent == "database":
            reply = await ask_database(update.message.text)
            await update.message.reply_text(reply, parse_mode='Markdown')

        elif intent == "action":
            action = await ask_action(update.message.text)

            if action["action"] == "toggle_heater":
                ok = await tuya_on(heater) if action["state"] else await tuya_off(heater)
                await update.message.reply_text(f"{'✅' if ok else '❌'} {action['reply']}")

            elif action["action"] == "toggle_fogger":
                ok = await tuya_on(fogger_fan) if action["state"] else await tuya_off(fogger_fan)
                await update.message.reply_text(f"{'✅' if ok else '❌'} {action['reply']}")

            elif action["action"] == "set_temp":
                TARGET_TEMP = float(action["value"])
                save_setting("target_temp", str(TARGET_TEMP))
                await update.message.reply_text(action["reply"], parse_mode='Markdown')

            elif action["action"] == "set_humidity":
                TARGET_HUMIDITY = float(action["value"])
                save_setting("target_humidity", str(TARGET_HUMIDITY))
                await update.message.reply_text(action["reply"], parse_mode='Markdown')

            elif action["action"] == "set_stage":
                GROWTH_STAGE = action["stage"]
                save_setting("growth_stage", GROWTH_STAGE)
                await update.message.reply_text(action["reply"], parse_mode='Markdown')

            elif action["action"] == "set_auto":
                AUTO_CONTROL = bool(action["state"])
                await update.message.reply_text(action["reply"])

            elif action["action"] == "schedule":
                device    = action["device"]
                state     = action["state"]
                delay_min = float(action["delay_minutes"])
                await update.message.reply_text(f"⏱️ {action['reply']}")

                async def delayed_action():
                    await asyncio.sleep(delay_min * 60)
                    if device == "heater":
                        ok = await tuya_on(heater) if state else await tuya_off(heater)
                    elif device == "fogger":
                        ok = await tuya_on(fogger_fan) if state else await tuya_off(fogger_fan)
                    else:
                        ok = False
                    label = "ON ✅" if state else "OFF ⚫"
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⏱️ Done: *{device.capitalize()}* → {label}",
                        parse_mode='Markdown'
                    )
                asyncio.create_task(delayed_action())

            elif action["action"] == "capture":
                await update.message.reply_text(action["reply"])
                filename, yolo_class, yolo_conf = await capture_and_detect(bot=context.bot)
                if filename:
                    caption = ""
                    if yolo_class:
                        stage   = YOLO_STAGE_MAP.get(yolo_class, "unknown")
                        caption = f"🔍 *{yolo_class.capitalize()}* ({yolo_conf:.1%}) → *{stage}*"
                    await update.message.reply_photo(
                        photo=open(filename, 'rb'),
                        caption=caption,
                        parse_mode='Markdown'
                    )
                    os.remove(filename)
                else:
                    await update.message.reply_text("❌ Failed to capture image")

            else:
                await update.message.reply_text(action["reply"])

        else:
            reply = await ask_chat(update.message.text, chat_id)
            await update.message.reply_text(reply, parse_mode='Markdown')

    except json.JSONDecodeError:
        await update.message.reply_text("⚠️ Couldn't parse that. Try rephrasing.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ─── STARTUP ───────────────────────────────────────────────────

async def post_init(application):
    global TARGET_TEMP, TARGET_HUMIDITY, GROWTH_STAGE

    s               = load_settings()
    TARGET_TEMP     = float(s.get("target_temp",     23.0))
    TARGET_HUMIDITY = float(s.get("target_humidity", 70.0))
    GROWTH_STAGE    = s.get("growth_stage", "colonisation")
    print(f"📂 Settings: {TARGET_TEMP}°C · {TARGET_HUMIDITY}% · {GROWTH_STAGE}")

    # Load YOLO in background — bot starts immediately, model loads silently
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_yolo_model)

    asyncio.create_task(auto_control_loop())
    asyncio.create_task(hourly_collection_task(application))

if __name__ == '__main__':
    init_db()
    s               = load_settings()
    TARGET_TEMP     = float(s.get("target_temp",     23.0))
    TARGET_HUMIDITY = float(s.get("target_humidity", 70.0))
    GROWTH_STAGE    = s.get("growth_stage", "colonisation")

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8703838581:AAFeFRkhbuErZpoDYiAPx_EzLZoAVtpvIEk")

    print("🍄 MushroomOS Starting...")
    print(f"🎯 Targets  : {TARGET_TEMP}°C · {TARGET_HUMIDITY}%")
    print(f"🌱 Stage    : {GROWTH_STAGE}")
    print(f"🧠 AI       : gemini-2.5-flash + gemini-2.5-flash-lite")
    print(f"🔍 YOLO     : {YOLO_MODEL_PATH}")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start",      start))
    app.add_handler(CommandHandler("help",       start))
    app.add_handler(CommandHandler("status",     status))
    app.add_handler(CommandHandler("pic",        pic))
    app.add_handler(CommandHandler("stage",      stage_command))
    app.add_handler(CommandHandler("data",       data_command))
    app.add_handler(CommandHandler("export",     export_data))
    app.add_handler(CommandHandler("auto_on",    auto_on))
    app.add_handler(CommandHandler("auto_off",   auto_off))
    app.add_handler(CommandHandler("heater_on",  heater_on))
    app.add_handler(CommandHandler("heater_off", heater_off))
    app.add_handler(CommandHandler("fogger_on",  fogger_on))
    app.add_handler(CommandHandler("fogger_off", fogger_off))
    app.add_handler(CommandHandler("clear",      clear_history))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_natural))

    print("🤖 Bot + Auto-control + Data Collection + Gemini AI + YOLO active\n")
    app.run_polling()
