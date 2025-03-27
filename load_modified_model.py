# load_modified_model.py

from ultralytics import YOLO
import torch

# Load the modified architecture that has two detection heads
# (one for the original 80 classes and one for the new "wallet" class).
# Ensure the YAML file (e.g. yolov8n-2xhead.yaml) has 'nc: 81'
model_modified = YOLO('ultralytics/cfg/models/v8/yolov8n-2xhead.yaml', task="detect").load('yolov8n.pt')
print("Modified model loaded with COCO pretrained weights.")

# Load the custom-trained wallet head weights from the file.
wallet_state_dict = torch.load("yolov8n_wallet_head.pth")
model_modified.load_state_dict(wallet_state_dict, strict=False)
print("Wallet head weights loaded into modified model.")

# Run inference on a test image.
# Replace "your_test_image.jpg" with the actual path to your test image.
results = model_modified("your_test_image.jpg")
results.show()  # This will open a window to display the detections.
