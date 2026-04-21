from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/fire_training/data.yaml", epochs=200)