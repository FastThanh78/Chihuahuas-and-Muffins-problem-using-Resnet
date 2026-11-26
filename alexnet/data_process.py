"""Dữ liệu cho AlexNet: dùng thẳng thư mục Processed của repo ResNet-18 (ảnh RGB, cạnh ngắn 256) và stats.json.

mean/std trong stats.json được repo ResNet-18 tính trên phần train (không gồm val), sau CenterCrop 224.
Cách chia val giống hệt repo đó, nên AlexNet và ResNet-18 có cùng train/val/test.
"""
from pathlib import Path

import torch

# Kaggle: dataset được gắn vào /kaggle/input; local: thư mục "processed data" nằm cạnh thư mục này,
# hoặc Data/Processed khi thư mục này nằm trong repo ResNet-18
SEARCH_FOLDERS = [Path("/kaggle/input"), Path(__file__).resolve().parent.parent / "processed data",
                  Path(__file__).resolve().parent.parent / "Data"]
found = [path for folder in SEARCH_FOLDERS if folder.exists() for path in folder.rglob("stats.json")]
assert found, "Không tìm thấy Processed/stats.json. Trên Kaggle: Add Input → dataset ảnh đã xử lý"

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
