import os
import json
import torch

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

RAW_FOLDER = "D:/Code/CS114 - ML/Project/Data/Raw"
PROCESSED_FOLDER = "D:/Code/CS114 - ML/Project/Data/Processed"
STATS_FILE = PROCESSED_FOLDER + "/stats.json"

SEED = 10
VAL_RATIO = 0.1
IMAGE_SIZE = 224


def save_processed_images():
    resize = transforms.Resize(256)

    for split in ["train", "test"]:
        for class_name in ["chihuahua", "muffin"]:
            raw_class_folder = RAW_FOLDER + "/" + split + "/" + class_name
            processed_class_folder = PROCESSED_FOLDER + "/" + split + "/" + class_name
            os.makedirs(processed_class_folder, exist_ok=True)

            for file_name in os.listdir(raw_class_folder):
                image = Image.open(raw_class_folder + "/" + file_name)
                image = image.convert("RGB")
                image = resize(image)
                image.save(processed_class_folder + "/" + file_name, quality=95)

            print(split, "-", class_name, ":", len(os.listdir(processed_class_folder)))


def split_train_val(number_of_images):
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(number_of_images, generator=generator).tolist()
    number_val = int(VAL_RATIO * number_of_images)
    train_indices = indices[number_val:]
    val_indices = indices[:number_val]
    return train_indices, val_indices


def compute_mean_std():
    transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(PROCESSED_FOLDER + "/train", transform=transform)
    train_indices, val_indices = split_train_val(len(dataset))
    train_dataset = Subset(dataset, train_indices)
    loader = DataLoader(train_dataset, batch_size=64, num_workers=4)

    total = torch.zeros(3)
    total_squared = torch.zeros(3)
    number_of_pixels = 0

    for images, labels in loader:
        total += images.sum(dim=(0, 2, 3))
        total_squared += (images ** 2).sum(dim=(0, 2, 3))
        number_of_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = total / number_of_pixels
    std = (total_squared / number_of_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    save_processed_images()

    mean, std = compute_mean_std()
    print("mean:", mean)
    print("std:", std)

    with open(STATS_FILE, "w") as file:
        json.dump({"mean": mean, "std": std}, file)

(64, 3, 7, 7)
(0, 1, 2, 3)
# 0 là số ảnh trong 1 patch
# 1 là số channel
# 2 là chiều rộng
# 3 là chiều cao
