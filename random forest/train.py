"""Random Forest chỉ dùng histogram màu HSV (đặc trưng ở random_forest.py).
Huấn luyện Random Forest theo khung train.py của repo ResNet-18: KHÔNG augmentation, KHÔNG dò tham số.
Ảnh chỉ được cắt giữa 224×224 (giống ResNet-18 và AlexNet), thu về 128×128, trích đặc trưng rồi cho rừng học.

Chạy: py -3.12 train.py
Ghi ra outputs/: random_forest.joblib (mô hình đã học), random_forest_test_predictions.csv (dự đoán từng ảnh test)
"""
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
from torchvision import datasets, transforms

from data_process import PROCESSED_FOLDER, IMAGE_SIZE, split_train_val
from random_forest import FEATURE_SIZE, build_model, extract_features

OUTPUT_FOLDER = Path(__file__).resolve().parent / "outputs"


def to_arrays(dataset):
    """Chạy transform trên từng ảnh → ma trận đặc trưng X và vector nhãn y.
    Một tiến trình (~5 ms/ảnh): trên Windows, khởi động 4 worker của DataLoader mất ~40 s, lâu hơn cả việc trích đặc trưng."""
    features, labels = zip(*(dataset[i] for i in range(len(dataset))))
    return np.stack(features), np.array(labels)


def load_data():
    # train, val, test dùng chung một phép biến đổi; không cần ToTensor/Normalize vì cây chỉ so ngưỡng
    transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.Resize(FEATURE_SIZE),
        extract_features,
    ])

    train_full = datasets.ImageFolder(PROCESSED_FOLDER + "/train", transform=transform)
    test_dataset = datasets.ImageFolder(PROCESSED_FOLDER + "/test", transform=transform)

    train_indices, val_indices = split_train_val(len(train_full))
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(train_full, val_indices)

    return to_arrays(train_dataset), to_arrays(val_dataset), to_arrays(test_dataset)


def fit(model, name, data):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    print("số ảnh × số đặc trưng | train", x_train.shape, "| val", x_val.shape, "| test", x_test.shape)

    start = time.time()
    model.fit(x_train, y_train)
    print(name, "fit", round(time.time() - start, 1), "s |",
          "train acc", round(model.score(x_train, y_train), 4), "|",
          "val acc", round(model.score(x_val, y_val), 4))

    probabilities = model.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)
    test_accuracy = float((predictions == y_test).mean())
    print(name, "test acc", round(test_accuracy, 4))
    print("confusion matrix (hàng = nhãn thật, cột = dự đoán; 0 = chihuahua, 1 = muffin):")
    print(confusion_matrix(y_test, predictions))

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_FOLDER / (name + ".joblib"), compress=3)

    # dự đoán từng ảnh test, cùng định dạng file của AlexNet để so sánh McNemar (1 = chihuahua)
    test_paths = [path for path, _ in datasets.ImageFolder(PROCESSED_FOLDER + "/test").samples]
    with open(OUTPUT_FOLDER / (name + "_test_predictions.csv"), "w") as file:
        file.write("key,label,prob_chihuahua,pred\n")
        for path, label, probability, prediction in zip(test_paths, y_test, probabilities[:, 0], predictions):
            key = os.path.relpath(path, PROCESSED_FOLDER).replace("\\", "/")
            file.write(f"{key},{int(label == 0)},{probability},{int(prediction == 0)}\n")
    return test_accuracy


def train_random_forest():
    start = time.time()
    data = load_data()
    print("trích đặc trưng", round(time.time() - start, 1), "s")
    return fit(build_model(), "random_forest", data)


if __name__ == "__main__":
    print("test accuracy:", train_random_forest())
