"""Dữ liệu cho Random Forest: dùng thẳng thư mục Processed của repo ResNet-18 (ảnh RGB, cạnh ngắn 256).

Cách chia val giống hệt repo đó (và alexnet_simple), nên RF, AlexNet và ResNet-18 có cùng train/val/test.
RF không dùng mean/std trong stats.json (cây chỉ so sánh ngưỡng); stats.json chỉ để tìm thư mục Processed.
"""
from pathlib import Path

import torch

# trong project: thư mục "processed data" ở phía trên thư mục này; trong repo ResNet-18: Data/Processed. Chỉ đọc, không ghi gì vào đó
found = [path for folder in Path(__file__).resolve().parents
         for path in [*(folder / "processed data").rglob("stats.json"), *(folder / "Data" / "Processed").glob("stats.json")]]
assert found, "Không tìm thấy processed data/.../Processed/stats.json hoặc Data/Processed/stats.json"

STATS_FILE = str(found[0])
PROCESSED_FOLDER = str(Path(STATS_FILE).parent)

SEED = 10
VAL_RATIO = 0.1
IMAGE_SIZE = 224


def split_train_val(number_of_images):
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(number_of_images, generator=generator).tolist()
    number_val = int(VAL_RATIO * number_of_images)
    train_indices = indices[number_val:]
    val_indices = indices[:number_val]
    return train_indices, val_indices
