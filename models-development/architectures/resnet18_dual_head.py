import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18DualHead(nn.Module):
    """
    ResNet18 with a dual-head regressor:
      main_head   -> predicts standard radiomic features (energy, mean, rms, …)
      ratio_head  -> predicts ratio/shape features (qcod, cov, skew, kurt) that are scale-invariant and historically hard to regress.

    The two heads share the same frozen backbone but learn independent
    projection layers, allowing different inductive biases.
    """

    # Features that go to the ratio head (order must match target_cols order)
    RATIO_FEATURES = {"stat_qcod", "stat_cov", "stat_skew", "stat_kurt"}

    def __init__(self, num_outputs: int, target_cols: list, in_channels: int = 6):
        super().__init__()

        # Backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            self.backbone.conv1.weight[:, 0:1, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            nn.init.kaiming_normal_(self.backbone.conv1.weight[:, 1:, :, :])

        # Freeze all
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in [self.backbone.conv1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            for param in layer.parameters():
                param.requires_grad = True

        self.backbone.fc = nn.Identity()

        self.target_cols = target_cols
        self.ratio_indices = [i for i, c in enumerate(target_cols) if c in self.RATIO_FEATURES]
        self.main_indices  = [i for i, c in enumerate(target_cols) if c not in self.RATIO_FEATURES]
        n_ratio = len(self.ratio_indices)
        n_main  = len(self.main_indices)

        # Main head (standard features)
        self.main_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, n_main),
        )

        # Ratio head (scale-invariant features)
        self.ratio_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, n_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                    # (B, 512)

        out_main  = self.main_head(features)           # (B, n_main)
        out_ratio = self.ratio_head(features)          # (B, n_ratio)

        # Re-assemble in original target_cols order
        batch = features.size(0)
        out = torch.empty(batch, len(self.target_cols), device=features.device, dtype=features.dtype)
        out[:, self.main_indices]  = out_main
        out[:, self.ratio_indices] = out_ratio
        return out

