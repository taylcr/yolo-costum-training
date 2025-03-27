from ultralytics import YOLO
import torch
import copy

# Load the pretrained model (COCO, 80 classes)
model = YOLO("yolov8n.pt")

# Optionally, keep a copy of the original state dict for comparison
old_dict = copy.deepcopy(model.state_dict())

# Train the model on your custom dataset.
# Freeze the first 22 layers so that only the head(s) get updated.
results = model.train(data="data.yaml", freeze=22, epochs=100, imgsz=640)
