"""Network architectures matching the trained weights."""

from typing import cast

import torch.nn as nn
from monai.networks.nets.segresnet_ds import SegResNetDS
from torchvision.models import efficientnet_b0


def efficientnet_b0_head(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return cast(nn.Module, model)


def segresnet() -> nn.Module:
    """SegResNet as built in the segmentation notebook (`volume_lib.build_model`).

    Four-channel (FLAIR, T1, T1CE, T2) input, three-channel (WT, TC, ET) logits; in eval mode it
    returns only the full-resolution output, without the deep-supervision heads.
    """
    return SegResNetDS(
        init_filters=32,
        in_channels=4,
        out_channels=3,
        norm="instance",
        blocks_down=(1, 2, 2, 4),
        dsdepth=3,
    )
