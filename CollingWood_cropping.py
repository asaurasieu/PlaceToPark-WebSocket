import cv2 
import json 
import os 

def cropping(image_folder, json_path, output_dir = "/Collingwood-Patches"
): 
    
    with open(json_path, 'r') as f: 
        spots = json.load(f)
        
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f'Found {len(image_files)} images in {image_folder}')
    
    for img_file in image_files: 
        image_path = os.path.join(image_folder, img_file)
        frame = cv2.imread(image_path)
        if frame is None: 
            print("Skipping, could not read files")
            continue 
        
        print("\nProcessing: {image_path}")
        
        for spot_info in spots: 
            spot_id = spot_info["id"]
            x1,y1 = spot_info["start"]
            x2,y2 = spot_info["end"]
            
            patch = frame[y1:y2, x1:x2]
            
            patch_resized = cv2.resize(patch, (150, 150), interpolation=cv2.INTER_AREA)
            
            cv2.imshow("Parking Spot", patch_resized)
            cv2.waitKey(1)
            
            label_input = input(f"Spot ID {spot_id} in {img_file} -> Enter 1 for occupied, 0 for available: ").strip()
            if label_input == "1":
                label = "occupied"
            elif label_input == "0":
                label = "available"
            else:
                print("Invalid input. Saving to 'unlabeled' folder.")
                label = "unlabeled"
            
            
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir): 
                os.makedirs(label_dir)
            
            patch_filename = f"{os.path.splitext(img_file)[0]}_spot_{spot_id}.jpg"
            patch_path = os.path.join(label_dir, patch_filename)
            
            cv2.imwrite(patch_path, patch_resized)
            print(f"Saved patch -> {patch_path}")
            
    cv2.destroyAllWindows()
    
folder_of_frames = "frames_for_patches"   
json_path = "grayscale_mask/parking_spots.json"           
output = "Collingwood-Patches"    

cropping(folder_of_frames, json_path, output_dir=output)