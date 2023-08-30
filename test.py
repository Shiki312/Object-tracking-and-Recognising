import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Ultralytics YOLOv8, the latest version of the acclaimed real-time object detection and image segmentation model.

# YOLOv8m model -- the medium model -- achieves a 50.2% mAP (mean average precision) when measured on COCO. When evaluated against Roboflow 100
# mAP campare ground truth bounding box to detected box and return a result.
model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# cap = cv2.VideoCapture('vidyolov8.mp4')
cap = cv2.VideoCapture('test1.mp4')
# cap = cv2.VideoCapture('test3.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count = 0
while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))
    if ret is None:
        break
    count += 1
    if count % 3 != 0:
        continue

    results = model.predict(frame)

    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    # print(px)
    for index, row in px.iterrows():
        # print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        # d is detected object which is identified by the class which is present in coco.txt file
        c = class_list[d]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(c), (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
