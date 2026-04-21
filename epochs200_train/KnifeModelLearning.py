from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/knife_training/data.yaml", epochs=250)