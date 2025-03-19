import cv2
import numpy as np
import json
import os

parking_spots = []
start_point = None
drawing = False

def draw_rectangle(event, x, y,frames, params):
    global parking_spots, start_point, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_image = image.copy()
        cv2.rectangle(temp_image, start_point, (x, y), (255, 255, 255), 2)
        cv2.imshow('Annotate Parking Spots', temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        parking_spots.append((start_point, end_point))
        drawing = False
        cv2.rectangle(image, start_point, end_point, (255, 255, 255), 2)
        cv2.imshow('Annotate Parking Spots', image)


image_path = 'edges_grayscale.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Could not load image:", image_path)
    exit()

cv2.imshow('Annotate Parking Spots', image)
cv2.setMouseCallback('Annotate Parking Spots', draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Annotated Parking Spots:", parking_spots)


mask = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')


total_spots = len(parking_spots)
if total_spots > 0:
    intensity_step = 255 // total_spots

for idx, (start, end) in enumerate(parking_spots):
    intensity = 255 - (idx * intensity_step) 
    cv2.rectangle(mask, start, end, intensity, -1)

mask_path = 'final_mask.png'
cv2.imwrite(mask_path, mask)
print(f"Grayscale mask saved to {mask_path}")

parking_data = [{"id": idx + 1, "start": start, "end": end} for idx, (start, end) in enumerate(parking_spots)]
json_path = 'parking_spots.json'
with open(json_path, 'w') as f:
    json.dump(parking_data, f)
print(f"Parking spots saved to {json_path}")
