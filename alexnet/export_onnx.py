"""Xuất trọng số tốt nhất (outputs/alexnet_scratch_best.pth) sang ONNX một file, rồi kiểm tra trên toàn bộ 1.184 ảnh test:
ONNX khớp PyTorch, và khớp file dự đoán của lần chạy Kaggle (outputs/alexnet_scratch_test_predictions.csv).

Chạy: py -3.12 alexnet_simple/export_onnx.py   (cần onnxscript, onnxruntime; trên Kaggle thiếu onnxscript nên bước này bị bỏ qua)
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from alexnet_scratch import AlexNet
from data_process import IMAGE_SIZE, PROCESSED_FOLDER
from train import load_data

OUT = Path(__file__).resolve().parent / "outputs"

if __name__ == "__main__":
    model = AlexNet(num_classes=2)
    model.load_state_dict(torch.load(OUT / "alexnet_scratch_best.pth", map_location="cpu"))
    model.eval()

    onnx_path = OUT / "alexnet_scratch.onnx"
    torch.onnx.export(
        model, (torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE),), str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", max=1024)}},
        external_data=False,  # mô hình 11 MB: gộp trọng số vào một file .onnx cho dễ chia sẻ
    )

    session = ort.InferenceSession(str(onnx_path))
    test_loader = load_data()[2]
    torch_logits, onnx_logits = [], []
    with torch.no_grad():
        for images, _ in test_loader:
            torch_logits.append(model(images).numpy())
            onnx_logits.append(session.run(None, {"image": images.numpy()})[0])
    torch_logits, onnx_logits = np.concatenate(torch_logits), np.concatenate(onnx_logits)

    # ImageFolder: chihuahua = 0, muffin = 1; file Kaggle dùng quy ước cũ (prob_chihuahua, nhãn 1 = chihuahua)
    prob_chihuahua = torch.softmax(torch.from_numpy(onnx_logits), dim=1)[:, 0].numpy()
    keys = [Path(path).relative_to(PROCESSED_FOLDER).as_posix() for path, _ in test_loader.dataset.samples]
    kaggle = pd.read_csv(OUT / "alexnet_scratch_test_predictions.csv").set_index("key").loc[keys]

    diff_torch = np.abs(torch_logits - onnx_logits).max()
    diff_kaggle = np.abs(prob_chihuahua - kaggle.prob_chihuahua.values).max()
    accuracy = ((prob_chihuahua >= 0.5).astype(int) == kaggle.label.values).mean()
    print(f"{onnx_path.name}: {onnx_path.stat().st_size / 1e6:.1f} MB")
    print(f"ONNX so với PyTorch (CPU): lệch logit tối đa {diff_torch:.2e}")
    print(f"ONNX so với lần chạy Kaggle (GPU): lệch p(chihuahua) tối đa {diff_kaggle:.2e} | test accuracy {accuracy:.4f}")
    assert diff_torch < 1e-3 and diff_kaggle < 1e-3, "ONNX không khớp"
