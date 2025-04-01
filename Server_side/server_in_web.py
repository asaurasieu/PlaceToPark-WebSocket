import asyncio
import websockets
import json
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
import yt_dlp
import time

# Model setup
device = torch.device('mps')
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("/Users/anita/Documents/ParkingProjectFlask/Final_models/model_version6.pth", map_location=device))
model.to(device)
model.eval()

# Constants
TOTAL_SPOTS = 25
CONFIDENCE_THRESHOLD = 0.80
FRAME_INTERVAL = 5
EMA_ALPHA = 0.2
EMA_THRESHOLD = 0.57
EMA_HIGH_THRESHOLD = 0.60
EMA_LOW_THRESHOLD = 0.54

# Image preprocessing setup
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load parking spots configuration
with open("/Users/anita/Documents/ParkingProjectFlask/grayscale_mask/parking_spots.json", "r") as f:
    parking_spots = json.load(f)

# Calculate original resolution from parking spots
max_x = max(max(spot["start"][0], spot["end"][0]) for spot in parking_spots)
max_y = max(max(spot["start"][1], spot["end"][1]) for spot in parking_spots)
original_width = max_x + 1
original_height = max_y + 1

# Global state
active_connections = set()
server_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
ema_values = {spot["id"]: 0.0 for spot in parking_spots}

def update_ema(spot_id, new_value):
    was_occupied = ema_values[spot_id] > EMA_THRESHOLD
    ema_values[spot_id] = EMA_ALPHA * new_value + (1 - EMA_ALPHA) * ema_values[spot_id]
    return ema_values[spot_id] > (EMA_LOW_THRESHOLD if was_occupied else EMA_HIGH_THRESHOLD)

async def process_frame(frame, scaled_spots):
    if frame is None or frame.size == 0:
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    occupied_count = 0
    spot_status = {}
    
    for spot in scaled_spots:
        spot_id = spot["id"]
        x_min, y_min = min(spot["start"][0], spot["end"][0]), min(spot["start"][1], spot["end"][1])
        x_max, y_max = max(spot["start"][0], spot["end"][0]), max(spot["start"][1], spot["end"][1])
        
        spot_image = rgb_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        if spot_image.shape[0] < 10 or spot_image.shape[1] < 10:
            spot_image = rgb_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        input_tensor = image_transform(Image.fromarray(spot_image)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            confidence = output.softmax(1)[0]
            confidence_score = float(max(confidence))
            is_occupied = bool(torch.max(output, 1)[1].item())
        
        current_value = 1.0 if (is_occupied and confidence_score >= CONFIDENCE_THRESHOLD) else 0.0
        is_occupied = update_ema(spot_id, current_value)
        
        spot_status[spot_id] = {
            "occupied": is_occupied,
            "confidence": f"{confidence_score:.2f}",
            "ema_value": f"{ema_values[spot_id]:.2f}"
        }
        
        if is_occupied:
            occupied_count += 1
    
    if active_connections:
        await asyncio.gather(*[conn.send(json.dumps({
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_spots": TOTAL_SPOTS,
            "occupied_spots": occupied_count,
            "detection_rate": f"{(occupied_count / TOTAL_SPOTS * 100):.2f}%",
            "spots": [{"id": spot_id, **status} for spot_id, status in spot_status.items()]
        })) for conn in active_connections])

async def handle_connection(websocket):
    print(f"New client connected from {websocket.remote_address}")
    active_connections.add(websocket)
    await websocket.send(json.dumps({"type": "session_start", "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")}))
    
    while True:
        if not await websocket.recv():
            break
    
    active_connections.remove(websocket)
    print(f"Client {websocket.remote_address} disconnected")

async def stream_capture():
    youtube_url = "https://www.youtube.com/live/EPKWu223XEg?si=js_LWhj38TicLhQY"
    ydl_opts = {'format': 'best[height<=1080]', 'quiet': True, 'no_warnings': True, 'socket_timeout': 30, 'retries': 10, 'ignoreerrors': True}
    
    while True:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if not info or 'url' not in info:
                await asyncio.sleep(1)
                continue
            
            cap = cv2.VideoCapture(info['url'])
            if not cap.isOpened():
                await asyncio.sleep(1)
                continue
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scaled_spots = [{
                "id": spot["id"],
                "start": [int(spot["start"][0] * width / original_width), int(spot["start"][1] * height / original_height)],
                "end": [int(spot["end"][0] * width / original_width), int(spot["end"][1] * height / original_height)]
            } for spot in parking_spots]
            
            frame_count = last_process = failures = 0
            
            while True:
                if not active_connections:
                    await asyncio.sleep(0.1)
                    continue
                
                if not cap.read()[0]:
                    if (failures := failures + 1) >= 5:
                        break
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                if frame_count == 1 or (current_time - last_process) >= FRAME_INTERVAL:
                    await process_frame(cap.read()[1], scaled_spots)
                    last_process = current_time
                
                await asyncio.sleep(0.01)
            
            cap.release()
            await asyncio.sleep(5)

async def main():
    server = await websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=10,
        close_timeout=10
    )
    
    print("WebSocket server started on ws://0.0.0.0:3000")
    capture_task = asyncio.create_task(stream_capture())
    await server.wait_closed()
    
    for ws in active_connections.copy():
        await ws.close()
    active_connections.clear()

asyncio.run(main()) 