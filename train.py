import json

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data_process import PROCESSED_FOLDER, STATS_FILE, IMAGE_SIZE, split_train_val


def load_data(batch_size=32):
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


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()

    print("train:", len(train_loader.dataset))
    print("val:", len(val_loader.dataset))
    print("test:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))
    print(images.shape)
