import cv2
from ultralytics import YOLO
import torch

def main():
    # Load the modified architecture that has two heads (expects 81 classes: 80 COCO + wallet)
    # Ensure the YAML file (e.g., yolov8n-2xhead.yaml) is configured with 'nc: 81'
    model = YOLO('yolov8n-2xhead.yaml', task='detect').load('yolov8n.pt')
    print("Modified model loaded with COCO weights.")
    
    # Load the extra wallet head weights (which were extracted and saved separately)
    extra_head_weights = torch.load('yolov8n_wallet_head.pt')
    model.load_state_dict(extra_head_weights, strict=False)
    print("Extra head weights loaded into the modified model.")

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
