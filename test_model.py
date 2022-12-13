import torch
from models.experimental import attempt_load

device = 'cuda' if torch.cuda.is_available else 'cpu'
half = device != 'cpu'

weights = r'H:\code\Python\YoloV7\yolov7\weights\yolov7.pt'
imgsz = 640


def load_model():
    with torch.no_grad():
        model = attempt_load(weights, map_location=device)
        if half:
            model.half()

        if device != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        return model
