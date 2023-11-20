from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    res = model(frame)

    for bounding_box in res:
        ann = Annotator(frame)

        boxes = bounding_box.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            ann.box_label(b, model.names[int(c)])

    frame = ann.result()

    cv2.imshow("Pothole Detection", frame)
    if (
        cv2.waitKey(1) & 0xFF == ord("q")
        or cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) == -1
    ):
        break

cap.release()
cv2.destroyAllWindows()