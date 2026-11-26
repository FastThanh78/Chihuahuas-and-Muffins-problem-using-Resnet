"""Huấn luyện AlexNet từ đầu, theo khung train.py của repo ResNet-18, nhưng KHÔNG augmentation:
ảnh chỉ được cắt giữa 224×224 rồi chuẩn hóa theo mean/std của train.

Chạy: python train.py
Ghi ra: alexnet_scratch_best.pth (val accuracy cao nhất), alexnet_scratch_history.csv, alexnet_scratch.onnx
"""
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from alexnet_scratch import AlexNet
from data_process import PROCESSED_FOLDER, STATS_FILE, IMAGE_SIZE, split_train_val


def load_data(batch_size=32):
    with open(STATS_FILE) as file:
        stats = json.load(file)

    # train, val, test dùng chung một phép biến đổi (ảnh trong Processed đã có cạnh ngắn 256)
    transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    train_full = datasets.ImageFolder(PROCESSED_FOLDER + "/train", transform=transform)
    test_dataset = datasets.ImageFolder(PROCESSED_FOLDER + "/test", transform=transform)

    train_indices, val_indices = split_train_val(len(train_full))
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(train_full, val_indices)

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


def fit(model, name, loaders, lr, epochs, device="cuda" if torch.cuda.is_available() else "cpu"):
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_path = name + "_best.pth"
    best_val_accuracy = 0
    history = []

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_function, device)
        scheduler.step()
        history.append((epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy))

        print(name, "epoch", epoch + 1, "|",
              "train loss", round(train_loss, 4), "acc", round(train_accuracy, 4), "|",
              "val loss", round(val_loss, 4), "acc", round(val_accuracy, 4))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_path)
            print("  saved best model")

    with open(name + "_history.csv", "w") as file:
        file.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")
        for row in history:
            file.write(",".join(str(value) for value in row) + "\n")

    model.load_state_dict(torch.load(best_path))
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
    print(name, "test loss", round(test_loss, 4), "acc", round(test_accuracy, 4))

    model.eval()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)  # batch phải > 1, nếu = 1 thì batch bị cố định
    try:  # cần gói onnxscript; nếu thiếu thì .pth và kết quả test ở trên vẫn còn nguyên
        torch.onnx.export(
            model, (dummy,), name + ".onnx",
            input_names=["image"], output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch", max=1024)}},
        )
    except Exception as error:
        print("bỏ qua xuất ONNX:", error)
    return test_accuracy


def train_scratch(epochs=30):
    torch.manual_seed(42)  # cố định khởi tạo trọng số và thứ tự xáo trộn ảnh
    return fit(AlexNet(num_classes=2), "alexnet_scratch", load_data(), lr=0.001, epochs=epochs)


if __name__ == "__main__":
    print("test accuracy:", train_scratch())
