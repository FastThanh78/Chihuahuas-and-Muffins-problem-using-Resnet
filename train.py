import json
import sys

import torch
from torch import nn

from resnet18_scratch import ResNet18

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data_process import PROCESSED_FOLDER, STATS_FILE, IMAGE_SIZE, split_train_val

from resnet_built_in import build_resnet, IMAGENET_MEAN, IMAGENET_STD


def load_data(batch_size=32, mean=None, std=None):
    if mean is None:
        with open(STATS_FILE) as file:
            stats = json.load(file)
        mean = stats["mean"]
        std = stats["std"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_full = datasets.ImageFolder(PROCESSED_FOLDER + "/train", transform=train_transform)
    val_full = datasets.ImageFolder(PROCESSED_FOLDER + "/train", transform=test_transform)
    test_dataset = datasets.ImageFolder(PROCESSED_FOLDER + "/test", transform=test_transform)

    train_indices, val_indices = split_train_val(len(train_full))
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.shape[0]

    return total_loss / total, correct / total

def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images.shape[0]
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.shape[0]

    return total_loss / total, correct / total



def fit(model, name, loaders, lr, epochs, device="cuda"):
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_path = name + "_best.pth"
    best_val_accuracy = 0

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_function, device)
        scheduler.step()

        print(name, "epoch", epoch + 1, "|",
                "train loss", round(train_loss, 4), "acc", round(train_accuracy, 4), "|",
                "val loss", round(val_loss, 4), "acc", round(val_accuracy, 4))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_path)
            print("  saved best model")

    model.load_state_dict(torch.load(best_path))
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
    print(name, "test loss", round(test_loss, 4), "acc", round(test_accuracy, 4))

    model.eval()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)  # batch phải > 1, nếu = 1 thì batch bị cố định
    torch.onnx.export(
        model, (dummy,), name + ".onnx",
        input_names=["image"], output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", max=1024)}},
    )
    return test_accuracy


def train_scratch(epochs=30):
    return fit(ResNet18(num_classes=2), "resnet18_scratch", load_data(), lr=0.001, epochs=epochs)


def finetune(name, epochs=10, lr=0.0001):
    loaders = load_data(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return fit(build_resnet(name), name + "_finetune", loaders, lr=lr, epochs=epochs)


if __name__ == "__main__":
    # python train.py                      fine-tune cả 3 model
    # python train.py resnet50             chỉ fine-tune resnet50
    # python train.py scratch resnet18     train từ đầu + fine-tune resnet18
    names = sys.argv[1:] or ["resnet18", "resnet34", "resnet50"]
    results = {}
    for name in names:
        if name == "scratch":
            results[name] = train_scratch() 
        else:
            finetune(name)
    print("test accuracy:", results)
