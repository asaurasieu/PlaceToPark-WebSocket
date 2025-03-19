import cv2
import numpy as np
import json
from ultralytics import YOLO

class PreciseCarDetector:
    def __init__(self, image_path):
        # Load image and prepare basic parameters
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        self.vis_image = self.image.copy()
        
        # Use larger, more accurate model
        self.model = YOLO("yolov8m.pt")  # Using extra-large model for better far-field detection
        
        # Detect initial cars with multiple confidence thresholds
        self.detect_cars()
        
        # Stores final annotated cars
        self.annotated_cars = []
        
    def detect_cars(self, conf_thresholds=[0.1, 0.05, 0.01]):
        """Detect cars using multiple confidence thresholds"""
        self.car_detections = []
        
        for conf_threshold in conf_thresholds:
            results = self.model.predict(source=self.image, conf=conf_threshold)
            
            # Filter car detections
            new_detections = [
                {
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    'conf': float(conf),
                    'center': [int((x1+x2)/2), int((y1+y2)/2)]
                }
                for box in results[0].boxes 
                if self.model.names[int(box.cls[0])] == "car"
                for conf, (x1, y1, x2, y2) in zip([float(box.conf[0])], [box.xyxy[0]])
            ]
            
            # Extend detections, avoiding duplicates
            self.car_detections.extend([
                det for det in new_detections 
                if not any(self.boxes_overlap(det['bbox'], existing['bbox']) 
                           for existing in self.car_detections)
            ])
        
    def get_nearest_box(self, x, y, dist_thresh=100):
        """Find the nearest car detection to the clicked point"""
        chosen_box = None
        min_dist = float("inf")

        for detection in self.car_detections:
            center_x, center_y = detection['center']
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < min_dist:
                min_dist = dist
                chosen_box = detection['bbox']

        if min_dist <= dist_thresh:
            return chosen_box
        return None

    def boxes_overlap(self, box1, box2, overlap_thresh=0.3):
        """Check if two boxes overlap significantly"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate overlap ratio
        overlap = intersection_area / min(box1_area, box2_area)
        return overlap > overlap_thresh

    def normalize_bbox_size(self, bbox):
        """Adjust bounding box size based on car's apparent size in the image"""
        x, y, w, h = bbox
        
        # Calculate relative size (cars further away appear smaller)
        # Use vertical position as a proxy for distance
        vertical_factor = 1 + (y / self.height)  # Adjust size based on vertical position
        
        # Standard car dimensions (adjust these based on your specific use case)
        target_width = max(80, int(w / vertical_factor))
        target_height = max(40, int(h / vertical_factor))
        
        # Center the box
        new_x = x + (w - target_width) // 2
        new_y = y + (h - target_height) // 2
        
        # Ensure within image bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        target_width = min(target_width, self.width - new_x)
        target_height = min(target_height, self.height - new_y)
        
        return [new_x, new_y, target_width, target_height]

    def refine_bbox_with_contours(self, bbox, click_point, margin=20):
        """
        Refine bounding box using local contour detection.
        The method crops a region around the rough bbox (extended by a margin),
        applies edge detection and finds contours. It then selects the contour
        that contains the click point and adjusts the bounding box accordingly.
        """
        x, y, w, h = bbox
        # Extend region by margin
        roi_x1 = max(0, x - margin)
        roi_y1 = max(0, y - margin)
        roi_x2 = min(self.width, x + w + margin)
        roi_y2 = min(self.height, y + h + margin)
        roi = self.image[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        # Dilate edges to close gaps
        edged = cv2.dilate(edged, None, iterations=1)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Adjust the click point relative to ROI coordinates
        roi_click = (click_point[0] - roi_x1, click_point[1] - roi_y1)
        best_contour = None
        min_distance = float('inf')
        
        for cnt in contours:
            # Get bounding rect for each contour
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            # Check if the click falls within the contour's bounding rect
            if (roi_click[0] >= cx and roi_click[0] <= cx + cw and
                roi_click[1] >= cy and roi_click[1] <= cy + ch):
                # Compute distance from click to the center of the contour
                center_rect = (cx + cw/2, cy + ch/2)
                dist = np.sqrt((roi_click[0] - center_rect[0])**2 + (roi_click[1] - center_rect[1])**2)
                if dist < min_distance:
                    min_distance = dist
                    best_contour = (cx, cy, cw, ch)
                    
        if best_contour is not None:
            refined_box = [roi_x1 + best_contour[0], roi_y1 + best_contour[1], best_contour[2], best_contour[3]]
            return refined_box
        
        # Return original bbox if no contour was found
        return bbox

    def click_event(self, event, x, y, flags, param):
        """Handle mouse click events for car detection and refine the bounding box."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Try to find a nearby car detection from YOLO
            box = self.get_nearest_box(x, y)

            if box is not None:
                # Normalize the bounding box size
                normalized_box = self.normalize_bbox_size(box)
                # Refine the bounding box using contour detection
                refined_box = self.refine_bbox_with_contours(normalized_box, (x, y))
                bx, by, bw, bh = refined_box
                print(f"Refined bounding box: {refined_box}")
            else:
                # Create a fallback bounding box if no detection found
                half_size = 50
                bx = max(0, x - half_size)
                by = max(0, y - half_size)
                bw = 100
                bh = 100

                # Ensure bounding box doesn't exceed image boundaries
                if bx + bw > self.width:
                    bw = self.width - bx
                if by + bh > self.height:
                    bh = self.height - by

                fallback_box = [bx, by, bw, bh]
                # Refine the fallback box
                refined_box = self.refine_bbox_with_contours(fallback_box, (x, y))
                bx, by, bw, bh = refined_box
                print("No YOLO detection near click. Created fallback refined box:", refined_box)

            # Draw the bounding box in green
            cv2.rectangle(self.vis_image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            # Draw the click as a red dot
            cv2.circle(self.vis_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click on Cars", self.vis_image)

            # Store the confirmed/fallback bounding box
            self.annotated_cars.append([bx, by, bw, bh])

    def run_interactive_detection(self):
        """Run interactive car detection"""
        # Create a window and show the original image
        cv2.namedWindow("Click on Cars", cv2.WINDOW_NORMAL)
        cv2.imshow("Click on Cars", self.vis_image)

        # Set the mouse callback
        cv2.setMouseCallback("Click on Cars", self.click_event)

        print("Click on the center of each car to create/confirm a bounding box.")
        print("Press 'q' to finish annotation.")

        # Wait for user interactions
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Close the window
        cv2.destroyAllWindows()

        print("Annotated bounding boxes:", self.annotated_cars)

        # Create and save the mask
        self.create_mask()
        self.save_annotations()

    def create_mask(self):
        """Create a gray mask with varying intensities"""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        total = len(self.annotated_cars)
        if total > 0:
            intensity_step = 255 // total
            for idx, (bx, by, bw, bh) in enumerate(self.annotated_cars):
                intensity = 255 - (idx * intensity_step)
                cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), intensity, -1)

        # Save mask
        cv2.imwrite("final_gray_mask.png", mask)

    def save_annotations(self):
        """Save bounding boxes to JSON"""
        parking_data = [
            {"id": idx + 1, "bbox": bbox}
            for idx, bbox in enumerate(self.annotated_cars)
        ]
        
        with open("parking_spots.json", "w") as f:
            json.dump(parking_data, f, indent=2)
        
        print(f"Bounding boxes saved to parking_spots.json")

def main():
    image_path = "frame_001.png"
    detector = PreciseCarDetector(image_path)
    detector.run_interactive_detection()

if __name__ == "__main__":
    main()
