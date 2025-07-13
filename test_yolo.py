from ultralytics import YOLO
import cv2

# Load YOLOv8n model (the smallest and fastest)
model = YOLO("yolov8n.pt")  # This will auto-download it

# Load a sample image (any JPG or PNG you have)
img = cv2.imread("test.jpg")  # Replace with your own image path

# Run prediction
results = model(img)
results[0].plot()

# Show result
cv2.imshow("YOLO Detection", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
