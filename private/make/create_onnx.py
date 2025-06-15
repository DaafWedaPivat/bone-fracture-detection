from ultralytics import YOLO

model = YOLO("../generated/yolov8l400/weights/best.pt")

model.export(format="onnx")



