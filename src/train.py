import argparse


def run():
    parser = argparse.ArgumentParser(description='Train settings')
    parser.add_argument('-m', type=str, help='Target model')
    args = parser.parse_args()

    m_str: str = args.m

    if not m_str:
        print('-m required')
    elif m_str.lower() == 'yolo':
        train_yolo()
    elif m_str.lower() == 'color':
        train_color()
    elif m_str.lower() == 'size':
        train_size()
    else:
        print('-m not supported')


def train_yolo():
    from ultralytics import YOLO

    model = YOLO('weights/yolo11n.pt')
    model.train(
        data='../resources/datasets/data.yaml',  # path to dataset YAML
        epochs=1,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4
    )
    model.export(format='pt')


def train_color():
    from ultralytics import YOLO

    model = YOLO('weights/yolo11n-cls.pt')
    model.train(
        data='../resources/simplified-car-color-dataset',
        epochs=20,  # number of training epochs
        imgsz=224,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=64
    )
    model.export()


def train_size():
    from ultralytics import YOLO

    model = YOLO('weights/yolo11n-cls.pt')
    model.train(
        data='../resources/simplified-car-size-dataset',
        epochs=25,  # number of training epochs
        imgsz=224,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=64
    )
    model.export()


if __name__ == '__main__':
    run()
