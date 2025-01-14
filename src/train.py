from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('yolo11n.pt')

    # Train the model
    train_results = model.train(
        data='../resources/datasets/data.yaml',  # path to dataset YAML
        epochs=1,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4
    )

    # Export the model to ONNX format
    path = model.export(format='onnx')  # return path to exported model
    print(path)
