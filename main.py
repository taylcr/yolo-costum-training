import cv2
from ultralytics import YOLO

# 1. Load your trained YOLOv8 model

#model = YOLO("yolov8n.pt")  # Replace 'best.pt' with your own weights file (e.g., 'yolov8n.pt', 'runs/detect/train/weights/best.pt', etc.)

model = YOLO("runs/detect/train8/weights/best.pt")


# 2. Open a connection to your webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # 3. Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break
    
    # 4. Perform inference
    results = model.predict(frame, conf=0.5)  # you can adjust the conf threshold as needed

    # 5. Draw the predictions on the frame
    # results[0].plot() returns an annotated NumPy array with bounding boxes, labels, etc.
    annotated_frame = results[0].plot()

    # 6. Show the annotated frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # 7. Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Release resources
cap.release()
cv2.destroyAllWindows()
