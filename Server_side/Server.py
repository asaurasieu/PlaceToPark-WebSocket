import asyncio
import websockets
import json
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import time
import os
from dotenv import load_dotenv
import yt_dlp


load_dotenv()

# -- Configuration Variables --
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))
MODEL_PATH = os.getenv("MODEL_PATH")
SPOTS_CONFIG_PATH = os.getenv("PARKING_SPOTS_CONFIG")
YOUTUBE_URL = os.getenv("YOUTUBE_URL")

TOTAL_SPOTS = int(os.getenv("TOTAL_SPOTS", 24))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 5))
EMA_ALPHA = float(os.getenv("EMA_ALPHA", 0.2))
EMA_THRESHOLD = float(os.getenv("EMA_THRESHOLD", 0.57))
EMA_HIGH_THRESHOLD = float(os.getenv("EMA_HIGH_THRESHOLD", 0.6))
EMA_LOW_THRESHOLD = float(os.getenv("EMA_LOW_THRESHOLD", 0.54))

# -- Device setup -- 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# -- Load the model --

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# -- Pytorch image processing -- 
transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -- Load parking spot coordinates from parking_spots.json --
with open(SPOTS_CONFIG_PATH, "r") as f:
    parking_spots = json.load(f)

# Each spot contains some coordinates that define the rectangle [start to end points] defining the area in the original image 
original_width = max(max(spot["start"][0], spot["end"][0]) for spot in parking_spots) + 1
original_height = max(max(spot["start"][1], spot["end"][1]) for spot in parking_spots) + 1

#  -- Track connected clients and start the spot detection --
active_clients = set()
# Initially all spots are assumed to be empty 
ema_values = {spot["id"]: 0.0 for spot in parking_spots}

# -- Start the EMA algorithm, which is used to smooth the detection over time, reducing false positives/negatives -- 
def smooth_occupancy(spot_id, new_value): 
    previous_value = ema_values[spot_id]
    was_occupied = previous_value > EMA_THRESHOLD
    ema_values[spot_id] = EMA_ALPHA * new_value + (1 - EMA_ALPHA) * previous_value
    threshold = EMA_LOW_THRESHOLD if was_occupied else EMA_HIGH_THRESHOLD 
    return ema_values[spot_id] > threshold

# -- Calculate ROI (Region of interest) for each image --
def calculate_roi(frame, spot): 
    x_start = min(spot["start"][0], spot["end"][0])
    y_start = min(spot["start"][1], spot["end"][1])
    x_end = max(spot["start"][0], spot["end"][0])
    y_end = max(spot["start"][1], spot["end"][1])
    return frame[int(y_start):int(y_end), int(x_start):int(x_end)]

# -- Process frames and detect occupancy -- 

def predict_spot(image): 
    input_tensor = transform_image(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad(): 
        output = model(input_tensor)
        probabilities = output.softmax(1)[0]
        confidence = float(probabilities.max())
        predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class == 1, confidence
    
async def analyze_frame(frame, scaled_spots): 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    occupied_count = 0
    spot_availability = {}
    
    for spot in scaled_spots: 
        spot_id = spot["id"]
        spot_image = calculate_roi(rgb_frame, spot)
        
        if spot_image.shape[0] < 10 or spot_image.shape[1] < 10: 
            continue 
        
        is_occupied_raw, confidence = predict_spot(spot_image)
        current_value = 1.0 if is_occupied_raw and confidence >= CONFIDENCE_THRESHOLD else 0.0
        is_occupied = smooth_occupancy(spot_id, current_value)
        
        spot_availability[spot_id] = not is_occupied
        
        if is_occupied: 
            occupied_count += 1
            
    if active_clients: 
        available_spots = TOTAL_SPOTS - occupied_count 
        data = {
            "available": available_spots,
            "total": TOTAL_SPOTS,
            "spots": [{"id": spot_id, "isAvailable": available} for spot_id, available in spot_availability.items()]
        }
        await asyncio.gather(*[client.send(json.dumps(data)) for client in active_clients])
        
# -- Client connection -- 
async def client_connection(websocket): 
    active_clients.add(websocket)
    try: 
        async for message in websocket: 
            if message == "close": 
                break
    finally: 
        active_clients.remove(websocket)
        
# -- Process the live video stream from YouTube -- 

async def capture_stream():
    stream_options = {
        'format': 'best[height<=1080]', 
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': 10,
        'ignoreerrors': True
    }

    while True:
        with yt_dlp.YoutubeDL(stream_options) as ydl:
            stream_info = ydl.extract_info(YOUTUBE_URL, download=False)

            if not stream_info or 'url' not in stream_info:
                await asyncio.sleep(1)  
                continue

            video = cv2.VideoCapture(stream_info['url'])
            if not video.isOpened():
                await asyncio.sleep(1)
                continue

            video.set(cv2.CAP_PROP_BUFFERSIZE, 10)

            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Scale parking spot coordinates to match current frame resolution
            scaled_spots = [{
                "id": spot["id"],
                "start": [int(spot["start"][0] * frame_width / original_width), int(spot["start"][1] * frame_height / original_height)],
                "end": [int(spot["end"][0] * frame_width / original_width), int(spot["end"][1] * frame_height / original_height)]
            } for spot in parking_spots]

            frame_count = 0
            last_processed = time.time()
            failed_reads = 0

            while True:
                if not active_clients:
                    await asyncio.sleep(0.1)
                    continue

                success, frame = video.read()
                if not success:
                    failed_reads += 1
                    if failed_reads >= 5:
                        break  
                    continue

                frame_count += 1
                now = time.time()

                if frame_count == 1 or (now - last_processed) >= FRAME_INTERVAL:
                    await analyze_frame(frame, scaled_spots)
                    last_processed = now

                await asyncio.sleep(0.01)

            video.release()
            await asyncio.sleep(5)
            

# -- Main function -- 
async def main(): 
    server = await websockets.serve(client_connection, HOST, PORT, ping_interval=20, ping_timeout=10, close_timeout=10)
    await asyncio.gather(server.wait_closed(), capture_stream())
    
# -- Start the server -- 
asyncio.run(main())
    
