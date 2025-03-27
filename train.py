from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch

# # Load a pretrained model for fine-tuning (pretrained on COCO, for example)
model = YOLO("yolov8n.pt")


# Use the model
#results = model.train(data="config.yaml", epochs=26)  # train the model

#results = model.train(data="config.yaml", epochs=26, freeze=[1, 10])

#results = model.train(data="config.yaml", epochs=26, freeze=10)

#results = model.train(data="config.yaml", epochs=26)

# Start fine-tuning with your dataset (new + original classes)
model.train(data='config.yaml', epochs=30, imgsz=640)