from ultralytics import YOLO

model = YOLO("../generated/yolov8l400_enhanced/weights/best.pt")

model.export(format="onnx")



