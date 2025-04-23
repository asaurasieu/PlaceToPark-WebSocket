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
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

# Get port from environment variable - Heroku will provide this
PORT = int(os.getenv('PORT', 8081))
HOST = os.getenv('HOST', '0.0.0.0')  # Use 0.0.0.0 to accept all incoming connections

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Model setup
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model_path = os.path.join(os.path.dirname(__file__), os.getenv('MODEL_PATH', 'Final_models/model_version6.pth'))
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
else:
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Constants from environment variables
total_spots = int(os.getenv('TOTAL_SPOTS', 24))
confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.80))
frame_interval = int(os.getenv('FRAME_INTERVAL', 5))
ema_alpha = float(os.getenv('EMA_ALPHA', 0.2))
ema_threshold = float(os.getenv('EMA_THRESHOLD', 0.57))
ema_high_threshold = float(os.getenv('EMA_HIGH_THRESHOLD', 0.60))
ema_low_threshold = float(os.getenv('EMA_LOW_THRESHOLD', 0.54))

# Image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load parking spots configuration
parking_spots_path = os.path.join(os.path.dirname(__file__), os.getenv('PARKING_SPOTS_CONFIG', 'grayscale_mask/parking_spots.json'))
if os.path.exists(parking_spots_path):
    with open(parking_spots_path, "r") as f:
        parking_spots = json.load(f)
    logger.info("Parking spots configuration loaded")
else:
    logger.error(f"Parking spots configuration not found at {parking_spots_path}")
    raise FileNotFoundError(f"Parking spots configuration not found at {parking_spots_path}")

# Initialize dimensions
max_x = max(max(spot["start"][0], spot["end"][0]) for spot in parking_spots)
max_y = max(max(spot["start"][1], spot["end"][1]) for spot in parking_spots)
original_width = max_x + 1
original_height = max_y + 1

# Connection management
active_connections = set()
server_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
ema_values = {spot["id"]: 0.0 for spot in parking_spots}

def update_ema(spot_id, new_value):
    was_occupied = ema_values[spot_id] > ema_threshold
    ema_values[spot_id] = ema_alpha * new_value + (1 - ema_alpha) * ema_values[spot_id]
    return ema_values[spot_id] > (ema_low_threshold if was_occupied else ema_high_threshold)

async def process_frame(frame, scaled_spots):
    if frame is None or frame.size == 0:
        return

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        occupied_count = 0
        spot_status = {}
        
        for spot in scaled_spots:
            spot_id = spot["id"]
            x_min, y_min = min(spot["start"][0], spot["end"][0]), min(spot["start"][1], spot["end"][1])
            x_max, y_max = max(spot["start"][0], spot["end"][0]), max(spot["start"][1], spot["end"][1])
            
            spot_image = rgb_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if spot_image.shape[0] < 10 or spot_image.shape[1] < 10:
                continue
            
            input_tensor = image_transform(Image.fromarray(spot_image)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                confidence = output.softmax(1)[0]
                confidence_score = float(max(confidence))
                is_occupied = bool(torch.max(output, 1)[1].item())
            
            current_value = 1.0 if (is_occupied and confidence_score >= confidence_threshold) else 0.0
            is_occupied = update_ema(spot_id, current_value)
            
            spot_status[spot_id] = {
                "occupied": is_occupied,
                "confidence": f"{confidence_score:.2f}",
                "ema_value": f"{ema_values[spot_id]:.2f}"
            }
            
            if is_occupied:
                occupied_count += 1

        if active_connections:
            message = json.dumps({
                "available": total_spots - occupied_count,
                "total": total_spots,
                "spots": [{"id": spot_id, "isAvailable": not status["occupied"]} for spot_id, status in spot_status.items()]
            })
            await asyncio.gather(*[conn.send(message) for conn in active_connections])
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}")

async def handle_connection(websocket):
    try:
        remote_address = websocket.remote_address
        logger.info(f"New client connected from {remote_address}")
        active_connections.add(websocket)
        
        while True:
            try:
                message = await websocket.recv()
                if not message:
                    break
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                break
                
    finally:
        active_connections.remove(websocket)
        logger.info(f"Client {remote_address} disconnected")

async def stream_capture():
    youtube_url = os.getenv('YOUTUBE_URL', 'https://www.youtube.com/live/EPKWu223XEg?si=js_LWhj38TicLhQY')
    ydl_opts = {
        'format': 'best[height<=1080]',
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': 10,
        'ignoreerrors': True
    }
    
    while True:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if not info or 'url' not in info:
                    logger.warning("Failed to get stream URL, retrying...")
                    await asyncio.sleep(1)
                    continue
                
                cap = cv2.VideoCapture(info['url'])
                if not cap.isOpened():
                    logger.warning("Failed to open video capture, retrying...")
                    await asyncio.sleep(1)
                    continue
                
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                scaled_spots = [{
                    "id": spot["id"],
                    "start": [int(spot["start"][0] * width / original_width),
                             int(spot["start"][1] * height / original_height)],
                    "end": [int(spot["end"][0] * width / original_width),
                           int(spot["end"][1] * height / original_height)]
                } for spot in parking_spots]
                
                frame_count = last_process = failures = 0
                
                while True:
                    if not active_connections:
                        await asyncio.sleep(0.1)
                        continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        if failures >= 5:
                            logger.warning("Too many failures, restarting capture...")
                            break
                        failures += 1
                        continue
                    
                    frame_count += 1
                    current_time = time.time()
                    
                    if frame_count == 1 or (current_time - last_process) >= frame_interval:
                        await process_frame(frame, scaled_spots)
                        last_process = current_time
                    
                    await asyncio.sleep(0.01)
                
                cap.release()
                
        except Exception as e:
            logger.error(f"Error in stream capture: {e}")
            
        await asyncio.sleep(5)

async def health_check():
    while True:
        try:
            logger.info(f"Active connections: {len(active_connections)}")
            logger.info(f"Server uptime: {datetime.now() - datetime.strptime(server_start_time, '%Y%m%d_%H%M%S')}")
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in health check: {e}")

async def main():
    try:
        logger.info(f"Starting WebSocket server on {HOST}:{PORT}")
        
        server = await websockets.serve(
            handle_connection,
            HOST,
            PORT,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        )
        
        logger.info(f"WebSocket server started - connect clients to ws://{HOST}:{PORT}")
        
        # Start background tasks
        capture_task = asyncio.create_task(stream_capture())
        health_check_task = asyncio.create_task(health_check())
        
        await server.wait_closed()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        for ws in active_connections.copy():
            await ws.close()
        active_connections.clear()

# Start the server
asyncio.run(main()) 