import torch
import torch.nn as nn


class SimpleCNN(nn.Module): #Compact CNN for 32x32 RGB images.
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential( #Convolutional feature extractor.
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential( #Project pooled features to class logits.
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None: #Use standard init schemes for supported layers.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor: #Extract features, then classify.
        return self.classifier(self.features(x))