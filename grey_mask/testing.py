import cv2 

mask_path = 'final_grey_mask.png'
test_image_path = 'frame_0012.jpg'

grey_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
test_image = cv2.imread(test_image_path)

if grey_mask.shape[:2] != test_image.shape[:2]:
    raise ValueError("The dimensions of the grey mask and the test image do not match")

grey_mask_colored = cv2.cvtColor(grey_mask, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(test_image, 0.7, grey_mask_colored, 0.4, 0)

cv2.imshow('Grey Mask on New Test Image', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_overlay_path = 'grey_mask_overlay.png'
cv2.imwrite(output_overlay_path, overlay)
print(f"Overlay saved to {output_overlay_path}")