import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train12/weights/best.pt")

results = model.predict("static_test/IMG_8905.jpg", conf=0.25, show=True)

# Extract the annotated image from the first result (adjust if needed)
annotated_img = results[0].plot()

# Create a resizable window and set a fixed size (e.g., 800x600)
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 800, 600)

# Display the image in your custom window
cv2.imshow('Detection', annotated_img)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()