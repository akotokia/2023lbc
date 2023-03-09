# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List

from .mixer import PointMixerEncoderLayer
from .transformer import PointTransformerEncoderLayer


_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


class Classifier(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        share_planes: int = 8,
        shrink_rate: int = 1,
        planes: List[int] = [32, 64, 128, 256],
        radii: List[int] = [2, 2, 2, 2],
        strides: List[int] = [1, 2, 2, 2],
        num_blocks: List[int] = [1, 1, 1, 1],
        algorithm: str = 'mixer',
    ) -> None:
        super().__init__()

        if algorithm == 'mixer':
            EncoderLayer = PointMixerEncoderLayer
        elif algorithm == 'transformer':
            EncoderLayer = PointTransformerEncoderLayer
        else:
            raise ValueError(algorithm)

        # Embedding
        modules = []

        in_features = in_planes
        out_features = planes[0]
        modules.append(nn.Linear(in_features, out_features, bias=False))
        modules.append(nn.BatchNorm1d(out_features))
        modules.append(nn.ReLU(inplace=True))

        self.add_module('embedding', nn.Sequential(*modules))

        # Encoder
        layers = []
        for i, config in enumerate(zip(planes, radii, strides, num_blocks)):
            in_features = out_features
            out_features = config[0]
            layers.append(EncoderLayer(
                in_features,
                out_features,
                share_planes,
                shrink_rate,
                config[1],
                config[2],
                config[3],
            ))

        self.add_module('encoder', nn.Sequential(*layers))

        # Classifier head
        modules = []

        in_features = out_features
        modules.append(nn.Linear(in_features, out_features, bias=False))
        modules.append(nn.BatchNorm1d(out_features))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.5))

        in_features = out_features
        out_features = in_features // 2
        modules.append(nn.Linear(in_features, out_features, bias=False))
        modules.append(nn.BatchNorm1d(out_features))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.5))

        in_features = out_features
        modules.append(nn.Linear(in_features, out_planes))

        self.add_module('head', nn.Sequential(*modules))

    def forward(self, points: Tensor) -> Tensor:
        # points.size: batch_size, n_points, 2
        batch_size, __, __ = points.size()

        # features.size: batch_size * n_points, 2
        features = torch.reshape(points, (-1, 2))

        # features.size: batch_size * n_points, mid_planes
        features = self.embedding(features)

        # outputs.size: batch_size * n_points', mid_planes
        __, outputs = self.encoder((points, features))
        # ----------->: batch_size, n_points', mid_planes
        __, mid_planes = outputs.size()
        outputs = outputs.view(batch_size, -1, mid_planes)
        # ----------->: batch_size, mid_planes, n_points'
        outputs = torch.transpose(outputs, 1, 2)
        # ----------->: batch_size, mid_planes, 1
        outputs = F.avg_pool1d(outputs, outputs.size(-1))
        # ----------->: batch_size, out_planes
        outputs = self.head(torch.squeeze(outputs, dim=2))
        return outputs
