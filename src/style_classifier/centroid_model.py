import torch
from torch import nn
from style_classifier_baselines.feature_based_baselines import utils as fbb_utils

from typing import Tuple, Callable, Optional


class CentroidModel(nn.Module):
    def __init__(self, backbone_and_transform: Tuple[nn.Module, Callable] | str,
                 centroids: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if isinstance(backbone_and_transform, str):
            backbone_and_transform = fbb_utils.get_model_and_transform(
                backbone_and_transform)
        self.backbone, self.transform = backbone_and_transform

        self.centroids: Optional[nn.Parameter] = None
        if centroids is not None:
            self.centroids = nn.Parameter(centroids, requires_grad=False)

    def load_centroids(self, centroids: torch.Tensor) -> None:
        self.centroids = nn.Parameter(centroids, requires_grad=True)

    def forward(self, x, apply_transform: bool = False) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("Centroids are not initialized.")

        if apply_transform:
            x = self.transform(x)

        x = self.backbone(x)

        # Find nearest centroid.
        x = -((x[:, None] - self.centroids[None]) ** 2).sum(-1)
        return x


"""
Easiest way to load the models:

import os
def load_centroid_model(model_dir: str) -> CentroidModel:
    model_args = torch.load(os.path.join(model_dir, "args.pt"))
    if model_args["arch"] != "CentroidModel":
        raise ValueError("Model is not a CentroidModel.")
    centroids = torch.load(os.path.join(model_dir, "centroids.pt"))

    return CentroidModel(model_args["bakbone"], centroids)
"""