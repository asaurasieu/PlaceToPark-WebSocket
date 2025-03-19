import os
import time
import json
import glob
from datetime import datetime
import cv2 
from ultralytics import YOLO


processed_files = set()
total_spots = 25  
detection_stats = {
    "total_frames": 0,
    "total_spots_expected": total_spots,
    "detection_rates": []
}


print("Loading YOLOv8 model...")
model = YOLO('/Users/anita/Desktop/Test/runs/detect/car_detection_model4/weights/best.pt')
print("Model loaded successfully.")


print("Loading parking spots data...")
with open("/Users/anita/Desktop/Test/grayscale_mask/parking_spots.json", "r") as f: 
    parking_spots = json.load(f)
print(f"Loaded {len(parking_spots)} parking spots.")

print("Starting monitoring for new frames...")


while True:
    current_files = set(glob.glob("frames/*.jpg"))
    new_files = current_files - processed_files
    
    for file_path in sorted(new_files):
        image = cv2.imread(file_path)
        if image is None: 
            print(f"Failed to read {file_path}")
            continue 
        
     
        results = model(file_path)
        result = results[0]
        
        spots_detected = len(result.boxes)
        
    
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            detections.append({
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "confidence": confidence,
                "class": class_id,
                "name": class_name
            })
        
       
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
            
            
            is_occupied = False
            for det in detections:
                center_x = (det["xmin"] + det["xmax"]) / 2
                center_y = (det["ymin"] + det["ymax"]) / 2
                if spot_xmin <= center_x <= spot_xmax and spot_ymin <= center_y <= spot_ymax:
                    is_occupied = True 
                    break 
            
            occupied_status[spot_id] = is_occupied
            if is_occupied:
                occupied_spots += 1
                
       
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
                
            cv2.rectangle(image, pt1, pt2, color, 2)
            cv2.putText(image, label, (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
       
        annotated_filename = f"annotated_{os.path.basename(file_path)}"
        cv2.imwrite(annotated_filename, image)
        
     
        cv2.imshow("Parking Spots", image)
        cv2.waitKey(1)
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "timestamp": timestamp,
            "image": os.path.basename(file_path),
            "spots_detected": spots_detected,
            "total_spots": total_spots,
            "occupied_spots": occupied_spots,
            "detection_rate": f"{detection_rate:.2f}%",
            "detections": detections
        }
        
        output_file = f"results_{os.path.basename(file_path).split('.')[0]}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=4)
        
       
        detection_stats["total_frames"] += 1
        detection_stats["detection_rates"].append(detection_rate)
        avg_detection_rate = sum(detection_stats["detection_rates"]) / len(detection_stats["detection_rates"])
        
        print(f"Processed {file_path}: {occupied_spots}/{total_spots} spots occupied ({detection_rate:.2f}%)")
        print(f"Average detection rate: {avg_detection_rate:.2f}%")
        
        
        processed_files.add(file_path)
            
   
    time.sleep(1)