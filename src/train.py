import argparse

from tqdm import tqdm

from datasets import ColorDataset


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
    else:
        print('-m not supported')


def train_yolo():
    from ultralytics import YOLO

    model = YOLO('yolo11n.pt')
    model.train(
        data='../resources/datasets/data.yaml',  # path to dataset YAML
        epochs=1,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4
    )
    model.export(format='pt')


def train_color():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms
    from torch.utils.data import DataLoader

    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ColorDataset(root_dir='../resources/car-color-dataset/train', transform=transform)
    val_dataset = ColorDataset(root_dir='../resources/car-color-dataset/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.resnet50()
    model.load_state_dict(torch.load('resnet50.pth'))
    model.fc = nn.Linear(model.fc.in_features, train_dataset.get_class_num())

    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}] started')

        model.train()
        train_loss = 0.
        train_correct = 0
        total = 0
        for images, labels in tqdm(train_loader, total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * train_correct / total

        model.eval()
        val_loss = 0.
        val_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, total=len(val_loader)):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100. * val_correct / total

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'weights/resnet50.pth')
            print('Best weights saved')

        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'End')


if __name__ == '__main__':
    run()
