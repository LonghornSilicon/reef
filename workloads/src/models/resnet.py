"""ResNet built from the from-scratch operator library.

Module and parameter names follow torchvision's ``ResNet`` exactly -- ``conv1``,
``bn1``, ``layer1``..``layer4``, ``fc``, and ``downsample.0``/``downsample.1``
inside a block -- so a torchvision ``state_dict`` loads with ``strict=True``.
"""

import torch
from torch import nn

from configs.resnet import RESNET_CONFIGS, STAGE_PLANES, ResNetConfig
from operators.activation import ReLU
from operators.convolution import Conv2d
from operators.linear import Linear
from operators.normalization import BatchNorm2d
from operators.pooling import AdaptiveAvgPool2d, MaxPool2d


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> Conv2d:
    """Build the padded 3x3 convolution used inside every residual block.

    Args:
        in_channels: Channels entering the convolution.
        out_channels: Channels leaving it.
        stride: Spatial stride.

    Returns:
        An unbiased 3x3 convolution that preserves the spatial size at
        ``stride == 1``.
    """
    return Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> Conv2d:
    """Build the 1x1 convolution used for projections and bottlenecks.

    Args:
        in_channels: Channels entering the convolution.
        out_channels: Channels leaving it.
        stride: Spatial stride.

    Returns:
        An unbiased 1x1 convolution.
    """
    return Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    """Two 3x3 convolutions and a residual connection, as in ResNet-18/34."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_eps: float = 1e-5,
    ) -> None:
        """Build the two convolutions and their norms.

        Args:
            in_channels: Channels entering the block.
            planes: Width of the block; also its output width.
            stride: Spatial stride, applied by the first convolution.
            downsample: Projection applied to the residual path when the
                shape changes, or ``None`` for an identity shortcut.
            norm_eps: Epsilon of both batch normalizations.
        """
        super().__init__()
        self.conv1 = conv3x3(in_channels, planes, stride)
        self.bn1 = BatchNorm2d(planes, eps=norm_eps)
        self.relu = ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, eps=norm_eps)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block over ``x``.

        Args:
            x: Tensor shaped ``(batch, in_channels, height, width)``.

        Returns:
            Tensor shaped ``(batch, planes, out_h, out_w)``.
        """
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """1x1, 3x3, 1x1 with a 4x expansion, as in ResNet-50/101.

    This is the torchvision "V1.5" ordering: the stride sits on the 3x3
    convolution, not on the leading 1x1.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_eps: float = 1e-5,
    ) -> None:
        """Build the three convolutions and their norms.

        Args:
            in_channels: Channels entering the block.
            planes: Bottleneck width; the block emits ``planes * 4``.
            stride: Spatial stride, applied by the 3x3 convolution.
            downsample: Projection applied to the residual path when the
                shape changes, or ``None`` for an identity shortcut.
            norm_eps: Epsilon of all three batch normalizations.
        """
        super().__init__()
        width = planes * self.expansion
        self.conv1 = conv1x1(in_channels, planes)
        self.bn1 = BatchNorm2d(planes, eps=norm_eps)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes, eps=norm_eps)
        self.conv3 = conv1x1(planes, width)
        self.bn3 = BatchNorm2d(width, eps=norm_eps)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block over ``x``.

        Args:
            x: Tensor shaped ``(batch, in_channels, height, width)``.

        Returns:
            Tensor shaped ``(batch, planes * 4, out_h, out_w)``.
        """
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


BLOCKS: dict[str, type[BasicBlock | Bottleneck]] = {
    "basic": BasicBlock,
    "bottleneck": Bottleneck,
}


class ResNet(nn.Module):
    """Residual image classifier matching torchvision layer for layer."""

    def __init__(self, config: ResNetConfig) -> None:
        """Assemble the stem, the four residual stages and the classifier.

        Args:
            config: Depth hyperparameters.
        """
        super().__init__()
        self.config = config
        block = BLOCKS[config.block]
        self.in_channels = config.stem_channels
        self.conv1 = Conv2d(
            3,
            config.stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = BatchNorm2d(config.stem_channels, eps=config.norm_eps)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        stages = [
            self._make_stage(
                block, planes, count, stride=1 if index == 0 else 2
            )
            for index, (planes, count) in enumerate(
                zip(STAGE_PLANES, config.blocks_per_stage, strict=True)
            )
        ]
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(config.feature_channels, config.num_classes)

    def _make_stage(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build one residual stage.

        Only the first block of a stage changes shape, so only it gets a
        projection shortcut; the rest keep the identity path.

        Args:
            block: Block class the stage is built from.
            planes: Width of the stage, before expansion.
            blocks: Number of blocks in the stage.
            stride: Spatial stride of the stage's first block.

        Returns:
            The stage as an ``nn.Sequential``.
        """
        eps = self.config.norm_eps
        out_channels = planes * block.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                BatchNorm2d(out_channels, eps=eps),
            )
        layers = [
            block(self.in_channels, planes, stride, downsample, norm_eps=eps)
        ]
        self.in_channels = out_channels
        layers.extend(
            block(self.in_channels, planes, norm_eps=eps)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: Tensor shaped ``(batch, 3, height, width)``.

        Returns:
            Logit tensor shaped ``(batch, num_classes)``.
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


def resnet(name: str) -> ResNet:
    """Build one of the published ResNet depths by name.

    Args:
        name: A key of :data:`~configs.resnet.RESNET_CONFIGS`, such as
            ``"ResNet-50"``.

    Returns:
        The corresponding model.

    Raises:
        KeyError: If ``name`` is not a known depth.
    """
    if name not in RESNET_CONFIGS:
        known = ", ".join(RESNET_CONFIGS)
        raise KeyError(f"unknown ResNet depth {name!r}; known depths: {known}")
    return ResNet(RESNET_CONFIGS[name])
