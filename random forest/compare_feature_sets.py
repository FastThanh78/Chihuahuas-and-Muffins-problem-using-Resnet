"""So sánh màu HSV với HOG trong CÙNG khung tối giản: rừng 300 cây mặc định (seed 42), cùng train/val/test.

Trích 1 lần HSV (256 cột, random_forest.py) và HOG (1.764 cột, hàm hog() dưới đây), rồi học 3 tổ hợp: HSV, HOG, HSV + HOG.
Kiểm tra chéo: tổ hợp HSV phải ra đúng file dự đoán đã lưu của train.py. Tốc độ trích đặc trưng đo trên 473 ảnh val
đã nạp sẵn trong RAM. Cột "MDI màu" = phần tổng mức giảm Gini do các cột HSV đóng góp.
Chạy: py -3.12 compare_feature_sets.py   → in bảng, ghi outputs/feature_sets.csv
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

from data_process import IMAGE_SIZE, PROCESSED_FOLDER, split_train_val
from random_forest import FEATURE_SIZE, build_model, extract_features

HERE = Path(__file__).resolve().parent
HOG_CELL, HOG_BINS = 16, 9
N_CELLS = FEATURE_SIZE // HOG_CELL                                                     # 8 × 8 cell
CELL_ID = (np.arange(FEATURE_SIZE) // HOG_CELL)[:, None] * N_CELLS + np.arange(FEATURE_SIZE) // HOG_CELL  # pixel → số thứ tự cell


def hog(gray):
    """HOG kiểu Dalal & Triggs: cell 16×16, 9 hướng 0–180°, block 2×2 cell, chuẩn hóa L2-Hys → 7 × 7 × 36 = 1.764 cột."""
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)                                     # kernel [-1, 0, 1]
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = ((angle % 180) // (180 / HOG_BINS)).astype(np.int64)                         # hướng không dấu
    cells = np.bincount((CELL_ID * HOG_BINS + bins).ravel(), magnitude.ravel(), N_CELLS * N_CELLS * HOG_BINS)
    cells = cells.reshape(N_CELLS, N_CELLS, HOG_BINS)
    blocks = np.concatenate([cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]], axis=-1)
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-10)
    blocks = np.minimum(blocks, 0.2)                                                    # L2-Hys: 1 cạnh rất mạnh không lấn át cả block
    return (blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-10)).ravel()


def extract_hog(image):
    return hog(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)).astype(np.float32)


edge = np.zeros((FEATURE_SIZE, FEATURE_SIZE), np.uint8)
edge[:, FEATURE_SIZE // 2:] = 255
assert hog(edge).reshape(-1, 4, HOG_BINS).sum((0, 1)).argmax() == 0                     # biên dọc → gradient ngang → bin 0–20°
assert hog(edge.T).reshape(-1, 4, HOG_BINS).sum((0, 1)).argmax() == HOG_BINS // 2       # biên ngang → bin 80–100°

EXTRACT = {"HSV": extract_features, "HOG": extract_hog}
SETS = ["HSV", "HOG", "HSV + HOG"]

crop = transforms.Compose([transforms.CenterCrop(IMAGE_SIZE), transforms.Resize(FEATURE_SIZE)])   # giống train.py, trừ bước cuối
train_full = datasets.ImageFolder(PROCESSED_FOLDER + "/train")
test_set = datasets.ImageFolder(PROCESSED_FOLDER + "/test")
train_indices, val_indices = split_train_val(len(train_full))
images = {name: [np.asarray(crop(dataset.loader(path))) for path, _ in dataset.samples]
          for name, dataset in (("train", train_full), ("test", test_set))}
X = {(group, split): np.stack([extract(image) for image in images[split]]) for group, extract in EXTRACT.items() for split in images}
y = {"train": np.array(train_full.targets), "test": np.array(test_set.targets)}                   # 0 = chihuahua

extract_ms = {}
val_images = [images["train"][i] for i in val_indices]
for group, extract in EXTRACT.items():
    for image in val_images[:20]:
        extract(image)                                                                  # khởi động trước khi bấm giờ
    start = time.perf_counter()
    for image in val_images:
        extract(image)
    extract_ms[group] = 1000 * (time.perf_counter() - start) / len(val_images)

rng = np.random.default_rng(42)
boot = [rng.integers(0, len(y["test"]), len(y["test"])) for _ in range(1000)]           # cùng mẫu bootstrap cho mọi tổ hợp
y_train, y_val = y["train"][train_indices], y["train"][val_indices]
rows, correct = [], {}
for name in SETS:
    groups = name.split(" + ")
    x_all = np.hstack([X[group, "train"] for group in groups])
    x_test = np.hstack([X[group, "test"] for group in groups])
    model = build_model()
    start = time.perf_counter()
    model.fit(x_all[train_indices], y_train)
    fit_s = time.perf_counter() - start
    p_val = model.predict_proba(x_all[val_indices])[:, 0]                               # cột 0 = chihuahua
    p_test = model.predict_proba(x_test)[:, 0]
    if name == "HSV":                                                                   # phải khớp mô hình đã lưu của train.py
        saved = pd.read_csv(HERE / "outputs" / "random_forest_test_predictions.csv")
        assert np.allclose(saved.prob_chihuahua, p_test)
    correct[name] = (p_test >= 0.5) == (y["test"] == 0)
    low, high = np.percentile([correct[name][i].mean() for i in boot], [2.5, 97.5])
    rows.append({"đặc trưng": name, "số cột": x_all.shape[1], "fit s": round(fit_s, 1),
                 "trích ms/ảnh": sum(extract_ms[group] for group in groups),
                 "MDI màu": model.feature_importances_[:256].sum() if "HSV" in groups else 0.0,
                 "val acc": ((p_val >= 0.5) == (y_val == 0)).mean(), "val AUC": roc_auc_score(y_val == 0, p_val),
                 "test acc": correct[name].mean(), "CI low": low, "CI high": high,
                 "test AUC": roc_auc_score(y["test"] == 0, p_test), "test sai": int((~correct[name]).sum())})

table = pd.DataFrame(rows).set_index("đặc trưng")
for name in SETS:                                                                       # McNemar chính xác so với HSV
    only_hsv = int((correct["HSV"] & ~correct[name]).sum())
    only_other = int((~correct["HSV"] & correct[name]).sum())
    table.loc[name, "chỉ HSV đúng"], table.loc[name, "chỉ tổ hợp đúng"] = only_hsv, only_other
    table.loc[name, "McNemar p"] = binomtest(only_hsv, only_hsv + only_other, 0.5).pvalue if only_hsv + only_other else 1.0
print(table.to_string(float_format=lambda v: f"{v:.4f}"))
(HERE / "outputs").mkdir(exist_ok=True)
table.to_csv(HERE / "outputs" / "feature_sets.csv", encoding="utf-8-sig")
