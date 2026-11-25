from torch import nn
from torchvision import models
import torch


RESNETS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet(name, num_classes=2):
    build, weights = RESNETS[name]
    model = build(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    for name in RESNETS:
        model = build_resnet(name).eval()
        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 2), out.shape
        print(name, "ok", out.shape)
