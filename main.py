from grabscreen import grab_screen
import cv2
import win32gui
import win32con
import torch
import numpy as np
from test_model import load_model
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import letterbox


classs = ["person", 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']


device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'

conf_thres = 0.50
iou_thres = 0.45

x, y = (1920, 1080)
re_x, re_y = (1920, 1080)

model = load_model()
imgsz = 640
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

while True:
    img0 = grab_screen(region=(0, 0, x, y))
    img0 = cv2.resize(img0, (re_x, re_y))

    img = letterbox(img0, imgsz, stride=stride)[0]

    img = img.transpose(2, 0, 1)[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.
    if len(img.shape) == 3:
        img = img[None]

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

    aims = []
    for i, det in enumerate(pred):
        s = ''
        s += '%gx%g ' % img.shape[2:]
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls in reversed(det):  # Write to file
            # bbox:(tag, x_center, y_center, x_width, y_width)
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh)  # label format
            aim = ('%g ' * len(line)).rstrip() % line
            aim = aim.split(' ')
            aims.append(aim)
    if len(aims):
        for i, det in enumerate(aims):
            tag, x_center, y_center, width, height = det
            x_center, width = re_x * float(x_center), re_x * float(width)
            y_center, height = re_y * float(y_center), re_y * float(height)
            top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
            bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
            color = (0, 255, 0)
            cv2.putText(img0, classs[int(tag)], top_left[0], top_left[1]-20, cv2.FONT_ITALIC, fontScale=2, color=color, thickness=3)
            cv2.rectangle(img0, top_left, bottom_right, color, thickness=3)

    cv2.namedWindow('test-window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test-window', re_x // 3, re_y // 3)
    cv2.imshow('test-window', img0)

    hwnd = win32gui.FindWindow(None, 'test-window')
    CVRECT = cv2.getWindowImageRect('test-window')
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
