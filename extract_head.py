# extract_head.py

import torch
from ultralytics import YOLO

# Load your trained model.
# Make sure you are using the weights from your fine-tuning run.
# For example, if your trained weights are saved in 'yolov8n.pt', use that.

#model = YOLO("yolov8n.pt")
model = YOLO("runs/detect/train12/weights/best.pt")


# Extract the weights for the extra head.
# In the patched model, the extra head's weights are stored with keys starting with "model.model.22".
# We need to rename them to "model.model.23" to match the modified architecture.
new_state_dict = {}
for key, value in model.state_dict().items():
    if key.startswith("model.model.22"):
        new_key = key.replace("model.model.22", "model.model.23")
        new_state_dict[new_key] = value

# Save the new head weights to a file.
torch.save(new_state_dict, "yolov8n_wallet_head.pt")
print("New head weights saved as 'yolov8n_wallet_head.pt'")
