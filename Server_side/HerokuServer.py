import os
import json
import asyncio
import logging
import time
from datetime import datetime

import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import websockets
import yt_dlp
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()))
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

MODEL_PATH = os.getenv("MODEL_PATH", "Final_models/model_version6.pth")
PARKING_CONFIG = os.getenv("PARKING_SPOTS_CONFIG", "grayscale_mask/parking_spots.json")
TOTAL_SPOTS = int(os.getenv("TOTAL_SPOTS", 24))
CONF_THRESH = float(os.getenv("CONFIDENCE_THRESHOLD", 0.80))
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 5))
EMA_ALPHA = float(os.getenv("EMA_ALPHA", 0.2))
EMA_THRESH = float(os.getenv("EMA_THRESHOLD", 0.57))
EMA_HIGH = float(os.getenv("EMA_HIGH_THRESHOLD", 0.60))
EMA_LOW = float(os.getenv("EMA_LOW_THRESHOLD", 0.54))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
if not os.path.isfile(MODEL_PATH):
    logger.error(f"Model not found: {MODEL_PATH}")
    raise FileNotFoundError(MODEL_PATH)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

with open(PARKING_CONFIG) as f:
    spots = json.load(f)
orig_w = max(s["end"][0] for s in spots) + 1
orig_h = max(s["end"][1] for s in spots) + 1

ema = {s["id"]: 0.0 for s in spots}
connections = set()
start_time = datetime.now()

def update_ema(spot_id, val):
    was = ema[spot_id] > EMA_THRESH
    ema[spot_id] = EMA_ALPHA * val + (1 - EMA_ALPHA) * ema[spot_id]
    return ema[spot_id] > (EMA_LOW if was else EMA_HIGH)

async def broadcast(message):
    if connections:
        await asyncio.wait([ws.send(message) for ws in connections])

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
        val = 1.0 if cls and conf >= CONF_THRESH else 0.0
        occupied = update_ema(s["id"], val)
        busy += int(occupied)
        statuses.append({"id": s["id"], "isAvailable": not occupied})
    msg = json.dumps({
        "available": TOTAL_SPOTS - busy,
        "total": TOTAL_SPOTS,
        "spots": statuses
    })
    await broadcast(msg)

async def capture():
    url = os.getenv("YOUTUBE_URL")
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
            if now - last >= FRAME_INTERVAL:
                await process(frame)
                last = now
            await asyncio.sleep(0.01)
        cap.release()

async def handler(ws, path):
    connections.add(ws)
    logger.info("Client connected")
    try:
        async for _ in ws:
            pass
    finally:
        connections.remove(ws)
        logger.info("Client disconnected")

async def health():
    while True:
        uptime = datetime.now() - start_time
        logger.info(f"Connections: {len(connections)}, Uptime: {uptime}")
        await asyncio.sleep(60)

async def main():
    server = await websockets.serve(handler, HOST, PORT)
    logger.info(f"Server running on ws://{HOST}:{PORT}")
    await asyncio.gather(capture(), health(), server.wait_closed())

asyncio.run(main())
