from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="C:/Users/happy/SilverBridgeAI/AISilverBridgeLJH/Data/Knife/data.yaml", epochs=500)