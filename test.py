import cv2
from ultralytics import YOLO

def main():

    # Load the YOLOv8 nano model (ensure 'yolov8n.pt' is available or adjust the path)
    #model = YOLO('yolov8n.pt')

    model = YOLO("runs/detect/train11/weights/best.pt")

    


    # Initialize webcam capture (0 is the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run YOLO detection on the frame
        results = model(frame)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
