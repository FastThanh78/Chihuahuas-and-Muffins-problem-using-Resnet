"""AlexNet huấn luyện từ đầu cho ảnh 3×224×224, 2 lớp (viết theo kiểu resnet18_scratch.py).

Giữ 5 lớp conv của AlexNet: kernel 11-5-3-3-3, số kênh 64-192-384-256-256, MaxPool 3×3 bước 2.
Hai thay đổi để học được trên 4.260 ảnh:
  - BatchNorm sau mỗi conv thay cho LRN;
  - Global Average Pooling + FC 256 → 512 → 256 → 2 thay cho khối FC 9216 → 4096 → 4096,
    giảm từ 61,1 triệu xuống 2,73 triệu tham số.
"""
import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  # BN đã có bias
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    return nn.Sequential(*layers)


class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64, kernel_size=11, stride=4, padding=2, pool=True),   # 64 × 27 × 27
            conv_block(64, 192, kernel_size=5, padding=2, pool=True),            # 192 × 13 × 13
            conv_block(192, 384, kernel_size=3, padding=1),                      # 384 × 13 × 13
            conv_block(384, 256, kernel_size=3, padding=1),                      # 256 × 13 × 13
            conv_block(256, 256, kernel_size=3, padding=1, pool=True),           # 256 × 6 × 6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = AlexNet().eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2), out.shape
    print("AlexNet ok", out.shape, "| tham số:", sum(p.numel() for p in model.parameters()))
