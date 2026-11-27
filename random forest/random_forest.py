"""Random Forest trên đặc trưng thủ công cho ảnh RGB 128×128, 2 lớp: CHỈ DÙNG MÀU HSV.

Mỗi ảnh → 1 vector 256 số: histogram HSV 16 tông × 4 bão hòa × 4 độ sáng (không giữ vị trí trong ảnh;
so sánh với HOG ở compare_feature_sets.py).
Rừng: 300 cây, còn lại giữ mặc định của scikit-learn (mỗi lần chia xét √256 = 16 đặc trưng, cây mọc hết cỡ).
"""
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURE_SIZE = 128
HSV_BINS = [16, 4, 4]


def color_histogram(rgb):
    """Tỷ lệ pixel rơi vào mỗi ô màu HSV (không phụ thuộc vị trí)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)                                         # OpenCV: H 0–179, S và V 0–255
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, [0, 180, 0, 256, 0, 256]).ravel()
    return histogram / histogram.sum()


def extract_features(image):
    """Ảnh RGB 128×128 (PIL hoặc numpy) → vector float32 256 số; là transform cuối cùng."""
    return color_histogram(np.asarray(image)).astype(np.float32)


def build_model():
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


if __name__ == "__main__":
    # đặc trưng phải phản ứng đúng lý thuyết trên ảnh tổng hợp
    size = FEATURE_SIZE
    red = np.zeros((size, size, 3), np.uint8)
    red[..., 0] = 255
    assert color_histogram(red).reshape(HSV_BINS)[0, -1, -1] == 1                       # đỏ thuần → H = 0, S và V cao nhất
    gray = np.full((size, size, 3), 128, np.uint8)
    assert color_histogram(gray).reshape(HSV_BINS)[0, 0, 2] == 1                        # xám → S = 0, V = 128 rơi vào mức thứ 3
    features = extract_features(np.random.default_rng(0).integers(0, 256, (size, size, 3), np.uint8))
    assert features.shape == (256,) and features.dtype == np.float32, features.shape
    assert abs(features.sum() - 1) < 1e-5                                                # tỷ lệ pixel, tổng = 1
    print("đặc trưng ok", features.shape)
