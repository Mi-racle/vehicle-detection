from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO('yolo11s.pt')

# Export the model to ONNX format
model.export(format='engine', imgsz=2048)  # creates 'yolo11s.onnx'
