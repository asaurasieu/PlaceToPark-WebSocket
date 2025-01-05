import cv2
import numpy as np
import json

# Initialize global variables
parking_spots = []
start_point = None
drawing = False

# Mouse callback function for drawing rectangles
def draw_rectangle(event, x, y,frames, params):
    global parking_spots, start_point, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a rectangle
        start_point = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Show the rectangle while dragging
        temp_image = image.copy()
        cv2.rectangle(temp_image, start_point, (x, y), (255, 255, 255), 2)
        cv2.imshow('Annotate Parking Spots', temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        # Finish the rectangle
        end_point = (x, y)
        parking_spots.append((start_point, end_point))
        drawing = False
        # Draw final rectangle on the image
        cv2.rectangle(image, start_point, end_point, (255, 255, 255), 2)
        cv2.imshow('Annotate Parking Spots', image)

# Load the image
image_path = 'frame_0007.jpg'  # Update with your image path
image = cv2.imread(image_path)

# Start annotation process
cv2.imshow('Annotate Parking Spots', image)
cv2.setMouseCallback('Annotate Parking Spots', draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save annotated parking spots
print("Annotated Parking Spots:", parking_spots)

# Create the grey mask
mask = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

# Calculate intensity gradient for grey tones
total_spots = len(parking_spots)
if total_spots > 0:
    intensity_step = 255 // total_spots  # Calculate the gradient step

# Draw parking spots on the mask
for idx, (start, end) in enumerate(parking_spots):
    intensity = 255 - (idx * intensity_step)  # Lighter grey for farther spots
    cv2.rectangle(mask, start, end, intensity, -1)  # Fill rectangle with grey tone

# Save the grey mask
mask_path = 'final_grey_mask.png'
cv2.imwrite(mask_path, mask)
print(f"Grey mask saved to {mask_path}")

# Save parking spot coordinates to a JSON file
parking_data = [{"id": idx + 1, "start": start, "end": end} for idx, (start, end) in enumerate(parking_spots)]
json_path = 'parking_spots.json'
with open(json_path, 'w') as f:
    json.dump(parking_data, f)
print(f"Parking spots saved to {json_path}")
