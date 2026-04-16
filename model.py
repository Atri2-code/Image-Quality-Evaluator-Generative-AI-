"""
model.py — Feature extraction backbone for perceptual quality scoring.
Uses pretrained ResNet-50 to extract deep feature representations.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualFeatureExtractor(nn.Module):
    """Extracts multi-scale features from ResNet-50 for quality assessment."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in resnet.parameters():
            p.requires_grad_(False)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        embedding = self.pool(f4).flatten(1)
        return {'layer1': f1, 'layer2': f2, 'layer3': f3, 'embedding': embedding}


class QualityScorer(nn.Module):
    """Lightweight MLP head that predicts a quality score from ResNet embeddings."""

    def __init__(self, input_dim=2048):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, embedding):
        return self.head(embedding).squeeze(-1) * 100
