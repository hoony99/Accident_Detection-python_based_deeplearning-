import torch
torch.cuda.empty_cache()
from ultralytics import YOLO


model = YOLO("/home/capstone_002_02/parameter/i1-yolov8s.pt")  
model.train(data="/home/capstone_002_02/data/data.yaml",  
            epochs=50,                   
            batch=2,               
            device='cuda')              

model.save("/home/capstone_002_02/parameter/yolo_model_1.pth")

results = model.evaluate(data="/home/capstone_002_02/data/data.yaml", imgsz=640, batch=2)
print('Model performance:', results)
