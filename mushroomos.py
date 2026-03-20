
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
import google.generativeai as genai

load_dotenv()  # reads from ~/.env

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# ─── CONFIG ────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash-lite")

# ─── HARDWARE ──────────────────────────────────────────────────
i2c    = board.I2C()
sensor = adafruit_sht31d.SHT31D(i2c, address=0x45)

heater = tinytuya.OutletDevice(
    dev_id=os.getenv("HEATER_ID"),
    address=os.getenv("HEATER_IP"),
    local_key=os.getenv("HEATER_KEY"),
    version=3.5
)
heater.set_socketPersistent(False)

fogger_fan = tinytuya.OutletDevice(
    dev_id=os.getenv("FOGGER_ID"),
    address=os.getenv("FOGGER_IP"),
    local_key=os.getenv("FOGGER_KEY"),
    version=3.5
)
fogger_fan.set_socketPersistent(False)

CAMERA_URL = 'http://192.168.1.33:8080/video'

# ─── DATABASE ──────────────────────────────────────────────────
conn   = sqlite3.connect('mushroom_data.db')
cursor = conn.cursor()
os.makedirs('images', exist_ok=True)

# ─── STATE ─────────────────────────────────────────────────────
hour_counter    = 0
start_time      = time.time()
TARGET_TEMP     = 23.0
TARGET_HUMIDITY = 70.0
AUTO_CONTROL    = True

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

# ─── CAMERA HELPER ─────────────────────────────────────────────

async def capture_image():
    """Capture a frame from camera, return filename or None."""
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
    return await loop.run_in_executor(None, _capture)

# ─── GEMINI AI ─────────────────────────────────────────────────

async def ask_ai(user_message: str) -> dict:
    temp, humidity = read_sensor()

    heater_s     = await tuya_status(heater)
    fogger_s     = await tuya_status(fogger_fan)
    heater_state = heater_s['dps']['1'] if heater_s else "unknown"
    fogger_state = fogger_s['dps']['1'] if fogger_s else "unknown"

    prompt = f"""You control a grey oyster mushroom grow chamber.

LIVE READINGS:
  Temperature : {f'{temp}°C' if temp else 'unavailable'}  (target: {TARGET_TEMP}°C)
  Humidity    : {f'{humidity}%' if humidity else 'unavailable'}  (target: {TARGET_HUMIDITY}%)

DEVICE STATES:
  Heater     : {'ON' if heater_state is True else 'OFF' if heater_state is False else heater_state}
  Fogger/Fan : {'ON' if fogger_state is True else 'OFF' if fogger_state is False else fogger_state}

AUTO-CONTROL: {'ON' if AUTO_CONTROL else 'OFF'}

Respond with ONLY a single JSON object — no markdown, no code fences, no extra text.

Valid actions:

Immediate device control:
{{"action":"toggle_heater","state":true,"reply":"Heater turned on"}}
{{"action":"toggle_fogger","state":false,"reply":"Fogger turned off"}}

Change targets:
{{"action":"set_temp","value":22.0,"reply":"Target temperature set to 22°C"}}
{{"action":"set_humidity","value":75.0,"reply":"Target humidity set to 75%"}}

Toggle auto-control:
{{"action":"set_auto","state":true,"reply":"Auto-control enabled"}}

Scheduled device action (use when user says "in X minutes/hours"):
{{"action":"schedule","device":"fogger","state":false,"delay_minutes":5,"reply":"Fogger will turn off in 5 minutes"}}
{{"action":"schedule","device":"heater","state":true,"delay_minutes":30,"reply":"Heater will turn on in 30 minutes"}}

Capture chamber image (use when user asks for photo, picture, image, pic, snapshot, camera):
{{"action":"capture","reply":"Capturing chamber image..."}}

Just answer a question:
{{"action":"none","reply":"Your answer here"}}

Rules:
- For scheduling, convert hours to minutes (1 hour = 60 minutes).
- Temp must stay between 15-28°C. Humidity between 60-98%.
- If user says take a pic, photo, picture, snap, camera, screenshot → always use capture action.
- One sentence reply max.

User message: {user_message}"""

    loop     = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: gemini.generate_content(prompt))
    raw      = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

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

async def hourly_collection_task():
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
                    cv2.imwrite(f'images/{image_filename}', frame)
                    return image_filename
                return None
            image_filename = await loop.run_in_executor(None, capture)

            cursor.execute('''
            INSERT INTO hourly_snapshots
            (timestamp, hours_since_start, temperature_c, humidity_pct,
             heater_on, fogger_fan_on, heater_power_w, fogger_power_w, image_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, hour_counter, temp, humidity,
                  heater_on, fogger_on, heater_power, fogger_power, image_filename))
            conn.commit()
            print(f"✅ Snapshot #{hour_counter} saved")
            hour_counter += 1

        except Exception as e:
            print(f"❌ Snapshot error: {e}")

        await asyncio.sleep(3600)

# ─── TELEGRAM COMMANDS ─────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍄 *MushroomOS Bot Active*\n\n"
        f"🤖 Auto-control: {'✅ ON' if AUTO_CONTROL else '❌ OFF'}\n"
        f"🎯 Target: {TARGET_TEMP}°C, {TARGET_HUMIDITY}%\n\n"
        "Commands:\n"
        "/status - Current readings\n"
        "/pic - Capture image\n"
        "/data - Recent snapshots\n"
        "/export - Download CSV\n"
        "/auto_on /auto_off - Toggle auto-control\n"
        "/heater_on /heater_off - Manual heater\n"
        "/fogger_on /fogger_off - Manual fogger\n\n"
        "💬 *Or just chat naturally:*\n"
        "_\"humidity is too high\"_\n"
        "_\"turn the heater off in 30 minutes\"_\n"
        "_\"take a pic\"_\n"
        "_\"what's the temperature?\"_",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        temp, humidity = read_sensor()
        heater_s = await tuya_status(heater)
        fogger_s = await tuya_status(fogger_fan)

        heater_on    = heater_s['dps']['1']       if heater_s else '?'
        fogger_on    = fogger_s['dps']['1']       if fogger_s else '?'
        heater_power = heater_s['dps']['19'] / 10 if heater_s else '?'
        fogger_power = fogger_s['dps']['19'] / 10 if fogger_s else '?'
        hours_running = int((time.time() - start_time) / 3600)

        await update.message.reply_text(
            f"🍄 *MushroomOS Status*\n\n"
            f"⏱️ Running: {hours_running} hours\n"
            f"📊 Snapshots: {hour_counter}\n"
            f"🤖 Auto-control: {'✅ ON' if AUTO_CONTROL else '❌ OFF'}\n\n"
            f"🌡️ Temperature: {temp}°C (target: {TARGET_TEMP}°C)\n"
            f"💧 Humidity: {humidity}% (target: {TARGET_HUMIDITY}%)\n\n"
            f"🔥 Heater: {'✅ ON' if heater_on is True else '❌ OFF'} ({heater_power}W)\n"
            f"💨 Fogger/Fan: {'✅ ON' if fogger_on is True else '❌ OFF'} ({fogger_power}W)",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def pic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("📸 Capturing...")
        filename = await capture_image()
        if filename:
            await update.message.reply_photo(photo=open(filename, 'rb'))
            os.remove(filename)
        else:
            await update.message.reply_text("❌ Failed to capture image")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def data_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cursor.execute("SELECT COUNT(*) FROM hourly_snapshots")
        count = cursor.fetchone()[0]
        if count == 0:
            await update.message.reply_text("📊 No data collected yet")
            return
        cursor.execute("""
        SELECT timestamp, temperature_c, humidity_pct, heater_on, fogger_fan_on
        FROM hourly_snapshots ORDER BY snapshot_id DESC LIMIT 5
        """)
        rows    = cursor.fetchall()
        message = f"📊 *Last {len(rows)} Snapshots* (Total: {count})\n\n"
        for row in rows:
            ts, temp, hum, heat, fog = row
            message += f"`{ts}` {temp:.1f}°C {hum:.1f}% H:{'✅' if heat else '❌'} F:{'✅' if fog else '❌'}\n"
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
                             'Heater_ON','Fogger_ON','Heater_W','Fogger_W','Image'])
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
    await update.message.reply_text("🤖 Auto-control ENABLED")

async def auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_CONTROL
    AUTO_CONTROL = False
    await update.message.reply_text("⚠️ Auto-control DISABLED")

async def heater_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_on(heater)
    await update.message.reply_text("🔥 Heater ON" if ok else "❌ Heater command failed")

async def heater_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_off(heater)
    await update.message.reply_text("❄️ Heater OFF" if ok else "❌ Heater command failed")

async def fogger_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_on(fogger_fan)
    await update.message.reply_text("💨 Fogger ON" if ok else "❌ Fogger command failed")

async def fogger_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await tuya_off(fogger_fan)
    await update.message.reply_text("💧 Fogger OFF" if ok else "❌ Fogger command failed")

# ─── NATURAL LANGUAGE ──────────────────────────────────────────

async def handle_natural(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TARGET_TEMP, TARGET_HUMIDITY, AUTO_CONTROL

    await context.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        action = await ask_ai(update.message.text)

        # ── Immediate device control ──
        if action["action"] == "toggle_heater":
            ok = await tuya_on(heater) if action["state"] else await tuya_off(heater)
            await update.message.reply_text(f"{'✅' if ok else '❌'} {action['reply']}")

        elif action["action"] == "toggle_fogger":
            ok = await tuya_on(fogger_fan) if action["state"] else await tuya_off(fogger_fan)
            await update.message.reply_text(f"{'✅' if ok else '❌'} {action['reply']}")

        # ── Target changes ──
        elif action["action"] == "set_temp":
            TARGET_TEMP = float(action["value"])
            await update.message.reply_text(f"✅ {action['reply']}")

        elif action["action"] == "set_humidity":
            TARGET_HUMIDITY = float(action["value"])
            await update.message.reply_text(f"✅ {action['reply']}")

        # ── Auto-control ──
        elif action["action"] == "set_auto":
            AUTO_CONTROL = bool(action["state"])
            await update.message.reply_text(f"✅ {action['reply']}")

        # ── Scheduled action ──
        elif action["action"] == "schedule":
            device     = action["device"]
            state      = action["state"]
            delay_min  = float(action["delay_minutes"])
            chat_id    = update.effective_chat.id

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
                    text=f"⏱️ Scheduled task done: {device.capitalize()} → {label}"
                )

            asyncio.create_task(delayed_action())

        # ── Capture image ──
        elif action["action"] == "capture":
            await update.message.reply_text("📸 Capturing...")
            filename = await capture_image()
            if filename:
                await update.message.reply_photo(photo=open(filename, 'rb'))
                os.remove(filename)
            else:
                await update.message.reply_text("❌ Failed to capture image")

        # ── Plain answer ──
        else:
            await update.message.reply_text(action["reply"])

    except json.JSONDecodeError:
        await update.message.reply_text("⚠️ Couldn't understand that. Try rephrasing.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ─── STARTUP ───────────────────────────────────────────────────

async def post_init(application):
    asyncio.create_task(auto_control_loop())
    asyncio.create_task(hourly_collection_task())

if __name__ == '__main__':
    print("🍄 MushroomOS Starting...")
    print(f"🎯 Targets: {TARGET_TEMP}°C, {TARGET_HUMIDITY}%")
    print(f"🤖 Auto-control: {'ON' if AUTO_CONTROL else 'OFF'}")
    print(f"🧠 AI: gemini-2.5-flash-lite")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start",      start))
    app.add_handler(CommandHandler("help",       start))
    app.add_handler(CommandHandler("status",     status))
    app.add_handler(CommandHandler("pic",        pic))
    app.add_handler(CommandHandler("data",       data_command))
    app.add_handler(CommandHandler("export",     export_data))
    app.add_handler(CommandHandler("auto_on",    auto_on))
    app.add_handler(CommandHandler("auto_off",   auto_off))
    app.add_handler(CommandHandler("heater_on",  heater_on))
    app.add_handler(CommandHandler("heater_off", heater_off))
    app.add_handler(CommandHandler("fogger_on",  fogger_on))
    app.add_handler(CommandHandler("fogger_off", fogger_off))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_natural))

    print("🤖 Bot + Auto-control + Data Collection + Gemini AI active")
    app.run_polling()
