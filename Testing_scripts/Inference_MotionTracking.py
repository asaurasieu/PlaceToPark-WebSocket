import os
import time
import json
import glob
from datetime import datetime
import cv2
import torch
from torchvision import transforms, models 
from PIL import Image
import numpy as np

processed_files = set()
total_spots = 25
detection_stats = {
    "total_frames": 0,
    "total_spots_expected": total_spots,
    "detection_rates": []
}


prev_frame = None
prev_features = None
prev_occupied_status = {}
tracking_points = {}

device = torch.device('mps')
print(f"Using device: {device}")

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  
model.load_state_dict(torch.load("/Users/anita/Documents/ParkingProjectFlask/Final_models/model_version6.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading parking spots data...")
with open("/Users/anita/Documents/ParkingProjectFlask/grayscale_mask/parking_spots.json", "r") as f:
    parking_spots = json.load(f)
print(f"Loaded {len(parking_spots)} parking spots.")

annotations_dir = "annotations"
detection_stats_dir = "detection_stats"
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(detection_stats_dir, exist_ok=True)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

def initialize_tracking_points(frame, parking_spots):
    tracking_points = {}
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for spot in parking_spots:
        spot_id = spot.get("id", f"Spot_{len(tracking_points)}")
        start = spot["start"]
        end = spot["end"]
        
        spot_xmin = min(start[0], end[0])
        spot_ymin = min(start[1], end[1])
        spot_xmax = max(start[0], end[0])
        spot_ymax = max(start[1], end[1])
        
        spot_mask = np.zeros_like(frame_gray)
        spot_mask[int(spot_ymin):int(spot_ymax), int(spot_xmin):int(spot_xmax)] = 255
        
        
        spot_features = cv2.goodFeaturesToTrack(
            frame_gray, 
            mask=spot_mask, 
            **feature_params
        )
        
        if spot_features is not None and len(spot_features) > 0:
            tracking_points[spot_id] = spot_features
        else:
            points = []
            step = 10
            for y in range(int(spot_ymin), int(spot_ymax), step):
                for x in range(int(spot_xmin), int(spot_xmax), step):
                    points.append([[float(x), float(y)]])
            tracking_points[spot_id] = np.array(points, dtype=np.float32)
    
    return frame_gray, tracking_points

def detect_movement(prev_gray, curr_gray, tracking_points):
    movement_scores = {}
    
    for spot_id, points in tracking_points.items():
        if len(points) == 0:
            movement_scores[spot_id] = 0
            continue
            
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None, **lk_params
        )
        
        if new_points is None:
            movement_scores[spot_id] = 0
            continue
        
       
        good_old = points[status == 1]
        good_new = new_points[status == 1]
        
      
        if len(good_old) > 0 and len(good_new) > 0:
            displacements = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
            movement_score = np.mean(displacements)
            movement_scores[spot_id] = movement_score
        else:
            movement_scores[spot_id] = 0
        
        tracking_points[spot_id] = good_new.reshape(-1, 1, 2)
    
    return movement_scores, tracking_points

print("Starting monitoring for new frames...")

while True:
    current_files = set(glob.glob("/Users/anita/Documents/ParkingProjectFlask/Server_side/frames/*.jpg"))
    new_files = current_files - processed_files
    
    for file_path in sorted(new_files):
        full_image = cv2.imread(file_path)
        if full_image is None:
            print(f"Failed to read {file_path}")
            continue
        
        curr_gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_gray, tracking_points = initialize_tracking_points(full_image, parking_spots)
            prev_frame = full_image.copy()
            movement_scores = {spot_id: 0 for spot_id in tracking_points.keys()}
        else:
            movement_scores, tracking_points = detect_movement(prev_gray, curr_gray, tracking_points)
            prev_gray = curr_gray.copy()
        
        full_image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        
        occupied_status = {}
        occupied_spots = 0
        
        for spot in parking_spots:
            spot_id = spot.get("id", f"Spot_{len(occupied_status)}")
            
            start = spot["start"]
            end = spot["end"]
            
            spot_xmin = min(start[0], end[0])
            spot_ymin = min(start[1], end[1])
            spot_xmax = max(start[0], end[0])
            spot_ymax = max(start[1], end[1])
            
            spot_image = full_image_rgb[int(spot_ymin):int(spot_ymax), int(spot_xmin):int(spot_xmax)]
            
            pil_image = Image.fromarray(spot_image)
            
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, prediction = torch.max(output, 1)
                is_occupied = bool(prediction.item())
            
            movement_score = movement_scores.get(spot_id, 0)
            movement_threshold = 1.0  
            
            if spot_id in prev_occupied_status:
                if movement_score > movement_threshold:
                    pass
                else:
                    confidence = output.softmax(1)[0]
                    if max(confidence) < 0.7:
                        is_occupied = prev_occupied_status[spot_id]
            
            occupied_status[spot_id] = is_occupied
            if is_occupied:
                occupied_spots += 1
        
        prev_occupied_status = occupied_status.copy()
        
        detection_rate = occupied_spots / total_spots * 100
        
        for spot in parking_spots:
            spot_id = spot.get("id", f"Spot_{len(occupied_status)}")
            
            start = spot["start"]
            end = spot["end"]
            
            spot_xmin = min(start[0], end[0])
            spot_ymin = min(start[1], end[1])
            spot_xmax = max(start[0], end[0])
            spot_ymax = max(start[1], end[1])
            
            pt1 = (int(spot_xmin), int(spot_ymin))
            pt2 = (int(spot_xmax), int(spot_ymax))
            
            if occupied_status.get(spot_id, False):
                color = (0, 0, 255)  
                label = f"Occupied {spot_id}"
            else:
                color = (0, 255, 0)  
                label = f"Empty {spot_id}"
                
            cv2.rectangle(full_image, pt1, pt2, color, 2)
            cv2.putText(full_image, label, (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
            if spot_id in tracking_points and tracking_points[spot_id] is not None:
                for point in tracking_points[spot_id]:
                    x, y = point.ravel()
                    cv2.circle(full_image, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        annotated_filename = os.path.join(annotations_dir, f"annotated_resnet_flow_{os.path.basename(file_path)}")
        cv2.imwrite(annotated_filename, full_image)
        
        cv2.imshow("Parking Spots - ResNet with Optical Flow", full_image)
        cv2.waitKey(1)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "timestamp": timestamp,
            "image": os.path.basename(file_path),
            "model": "CustomModel_OpticalFlow",
            "total_spots": total_spots,
            "occupied_spots": occupied_spots,
            "detection_rate": f"{detection_rate:.2f}%",
            "movement_detected": {spot_id: f"{score:.2f}" for spot_id, score in movement_scores.items() if score > 0.5}
        }
        
        output_file = os.path.join(detection_stats_dir, f"results_flow_{os.path.basename(file_path).split('.')[0]}.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=4)
        
        detection_stats["total_frames"] += 1
        detection_stats["detection_rates"].append(detection_rate)
        avg_detection_rate = sum(detection_stats["detection_rates"]) / len(detection_stats["detection_rates"])
        
        print(f"Processed {file_path}: {occupied_spots}/{total_spots} spots occupied ({detection_rate:.2f}%)")
        print(f"Average detection rate: {avg_detection_rate:.2f}%")
        print(f"Movement detected in {len([s for s,v in movement_scores.items() if v > 0.5])} spots")
        
        processed_files.add(file_path)
            
    time.sleep(1)