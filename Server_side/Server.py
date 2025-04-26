import os
import json
import asyncio
import logging
import time
from datetime import datetime
from aiohttp import web

import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import yt_dlp
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=getattr(logging, os.getenv("log_level", "INFO").upper()))
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "8081"))
HOST = "0.0.0.0"

model_path = os.getenv("model_path", "Final_model/model_version6.pth")
parking_config = os.getenv("parking_spots_config", "grayscale_mask/parking_spots.json")
total_spots = int(os.getenv("total_spots", 24))
conf_thresh = float(os.getenv("confidence_threshold", 0.80))
frame_interval = int(os.getenv("frame_interval", 5))
ema_alpha = float(os.getenv("ema_alpha", 0.2))
ema_thresh = float(os.getenv("ema_threshold", 0.57))
ema_high = float(os.getenv("ema_high_threshold", 0.60))
ema_low = float(os.getenv("ema_low_threshold", 0.54))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
if not os.path.isfile(model_path):
    logger.error(f"Model not found: {model_path}")
    raise FileNotFoundError(model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

if not os.path.isfile(parking_config):
    logger.error(f"Parking config not found: {parking_config}")
    raise FileNotFoundError(parking_config)
with open(parking_config) as f:
    spots = json.load(f)
orig_w = max(s["end"][0] for s in spots) + 1
orig_h = max(s["end"][1] for s in spots) + 1

ema = {s["id"]: 0.0 for s in spots}
connections = set()
start_time = datetime.now()

def update_ema(spot_id, val):
    was = ema[spot_id] > ema_thresh
    ema[spot_id] = ema_alpha * val + (1 - ema_alpha) * ema[spot_id]
    return ema[spot_id] > (ema_low if was else ema_high)

async def broadcast(message):
    if connections:
        await asyncio.wait([ws.send_str(message) for ws in connections])

async def process(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    busy = 0
    statuses = []
    for s in spots:
        x1 = int(s["start"][0] * w / orig_w)
        y1 = int(s["start"][1] * h / orig_h)
        x2 = int(s["end"][0]   * w / orig_w)
        y2 = int(s["end"][1]   * h / orig_h)
        crop = rgb[y1:y2, x1:x2]
        if min(crop.shape[:2]) < 10:
            continue
        inp = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            probs = out.softmax(1)[0]
            cls = int(probs.argmax())
            conf = float(probs[cls])
        val = 1.0 if cls and conf >= conf_thresh else 0.0
        occupied = update_ema(s["id"], val)
        busy += int(occupied)
        statuses.append({"id": s["id"], "isAvailable": not occupied})
    msg = json.dumps({
        "available": total_spots - busy,
        "total": total_spots,
        "spots": statuses
    })
    await broadcast(msg)

async def capture():
    url = os.getenv("youtube_url")
    ydl_opts = {'format': 'best[height<=1080]', 'quiet': True}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    while True:
        info = ydl.extract_info(url, download=False)
        stream_url = info.get("url")
        cap = cv2.VideoCapture(stream_url or "")
        if not cap.isOpened():
            await asyncio.sleep(5)
            continue
        last = time.time()
        failures = 0
        while True:
            if not connections:
                await asyncio.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                failures += 1
                if failures > 5:
                    break
                continue
            now = time.time()
            if now - last >= frame_interval:
                await process(frame)
                last = now
            await asyncio.sleep(0.01)
        cap.release()

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connections.add(ws)
    logger.info("WebSocket client connected")
    try:
        async for msg in ws:
            pass  # Handle incoming messages if needed
    finally:
        connections.remove(ws)
        logger.info("WebSocket client disconnected")
    return ws

async def health_check(request):
    return web.Response(text="OK")

app = web.Application()
app.router.add_get('/health', health_check)
app.router.add_get('/ws', websocket_handler)

async def on_startup(app):
    app['capture_task'] = asyncio.create_task(capture())

app.on_startup.append(on_startup)

web.run_app(app, host=HOST, port=PORT)
