# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Tuple

from .utils import circular_grouping_index


_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


class TransitionDown(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        radius: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.radius = radius
        self.stride = stride

        # Modules
        modules = []
        modules.append(nn.Linear(2 + in_planes, out_planes, bias=False))
        modules.append(nn.BatchNorm1d(out_planes))
        modules.append(nn.ReLU(inplace=True))

        self.add_module('modules', nn.Sequential(*modules))

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # points.size: batch_size, n_points, 2
        # features.size: batch_size * n_points, in_planes
        points, features = inputs

        __, in_planes = features.size()
        batch_size, n_points, __ = points.size()
        n_samples = self.radius * 2

        # indexes.size: batch_size, n_points, n_samples
        indexes = circular_grouping_index(points, self.radius)
        # ----------->: batch_size * n_points * n_samples
        indexes = torch.flatten(indexes.to(points.device))

        # nn_points.size: batch_size * n_points, 2
        nn_points = points.view(-1, 2)
        # ------------->: batch_size * n_points * n_samples, 2
        nn_points = nn_points[indexes]
        # ------------->: batch_size, n_points, n_samples, 2
        nn_points = nn_points.view(batch_size, n_points, n_samples, -1)

        # translation.size: batch_size, n_points, n_samples, 2
        translation = nn_points - torch.unsqueeze(points, 2)
        # --------------->: batch_size, n_points / stride, n_samples, 2
        translation = translation[:, ::self.stride]
        # --------------->: batch_size * (n_points / stride) * n_samples, 2
        translation = torch.reshape(translation, (-1, 2))

        # nn_features.size: batch_size * n_points * n_samples, in_planes
        nn_features = features[indexes]
        # --------------->: batch_size, n_points, n_samples, in_planes
        nn_features = nn_features.view(batch_size, n_points, n_samples, -1)
        # --------------->: batch_size, n_points / stride, n_samples, in_planes
        nn_features = nn_features[:, ::self.stride]
        # --------------->: batch_size * (n_points / stride) * n_samples, in_planes
        nn_features = torch.reshape(nn_features, (-1, in_planes))

        # outputs.size: batch_size * (n_points / stride) * n_samples, out_planes
        outputs = self.modules(torch.cat((translation, nn_features), dim=1))
        # ----------->: batch_size * n_points / stride, n_samples, out_planes
        __, out_planes = outputs.size()
        outputs = outputs.view(-1, n_samples, out_planes)
        # ----------->: batch_size * n_points / stride, out_planes, n_samples
        outputs = torch.transpose(outputs, 1, 2)
        # ----------->: batch_size * n_points / stride, out_planes, 1
        outputs = F.max_pool1d(outputs, outputs.size(-1))
        # ----------->: batch_size * n_points / stride, out_planes
        outputs = torch.squeeze(outputs, dim=2)

        # points.size: batch_size, n_points / stride, 2
        points = points[:, ::self.stride]

        return points, outputs


class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        share_planes: int,
        shrink_rate: int,
        radius: int,
    ) -> None:
        super().__init__()

        mid_planes = out_planes // shrink_rate

        self.share_planes = share_planes
        self.radius = radius

        self.add_module('to_q', nn.Linear(in_planes, mid_planes))
        self.add_module('to_k', nn.Linear(in_planes, mid_planes))
        self.add_module('to_v', nn.Linear(in_planes, out_planes))

        # Position encoding
        modules = []
        modules.append(nn.Linear(2, 2, bias=False))
        modules.append(nn.BatchNorm1d(2))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(2, out_planes))

        self.add_module('to_p', nn.Sequential(*modules))

        # Attention
        modules = []

        in_features = mid_planes
        modules.append(nn.BatchNorm1d(in_features))
        modules.append(nn.ReLU(inplace=True))

        out_features = mid_planes // share_planes
        modules.append(nn.Linear(in_features, out_features, bias=False))
        modules.append(nn.BatchNorm1d(out_features))
        modules.append(nn.ReLU(inplace=True))

        in_features = out_features
        out_features = out_planes // share_planes
        modules.append(nn.Linear(in_features, out_features))
        modules.append(nn.Softmax(dim=1))

        self.add_module('to_w', nn.Sequential(*modules))

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        # points.size: batch_size, n_points, 2
        # features.size: batch_size * n_points, in_planes
        points, features = inputs

        batch_size, n_points, __ = points.size()
        n_samples = self.radius * 2
        share_planes = self.share_planes

        # indexes.size: batch_size, n_points, n_samples
        indexes = circular_grouping_index(points, self.radius)
        # ----------->: batch_size * n_points * n_samples
        indexes = torch.flatten(indexes.to(points.device))

        # points.size: batch_size * n_points, 2
        points = points.view(-1, 2)

        # nn_points.size: batch_size * n_points * n_samples, 2
        nn_points = points[indexes]
        # ------------->: batch_size * n_points, n_samples, 2
        nn_points = nn_points.view(-1, n_samples, 2)

        # translation.size: batch_size * n_points, n_samples, 2
        translation = nn_points - torch.unsqueeze(points, 1)
        # --------------->: batch_size * n_points * n_samples, 2
        translation = translation.view(-1, 2)

        # r.size: batch_size * n_points * n_samples, out_planes
        r = self.to_p(translation)

        # q.size: batch_size * n_points, mid_planes
        # k.size: batch_size * n_points, mid_planes
        # v.size: batch_size * n_points, out_planes
        q = self.to_q(features)
        k = self.to_k(features)
        v = self.to_v(features)

        mid_planes = k.size(-1)
        out_planes = v.size(-1)

        # k.size: batch_size * n_points * n_samples, mid_planes
        k = k[indexes]
        # ----->: batch_size * n_points, n_samples, mid_planes
        k = k.view(-1, n_samples, mid_planes)

        # v.size: batch_size * n_points * n_samples, out_planes
        v = v[indexes]
        v = v + r

        # r.size: batch_size * n_points, n_samples, out_planes / mid_planes, mid_planes
        r = r.view(-1, n_samples, out_planes // mid_planes, mid_planes)
        # ----->: batch_size * n_points, n_samples, mid_planes
        r = torch.sum(r, dim=2, keepdim=False)

        # w.size: batch_size * n_points, n_samples, mid_planes
        w = r + k - torch.unsqueeze(q, 1)
        # ----->: batch_size * n_points * n_samples, out_planes / share_planes
        w = self.to_w(w.view(-1, mid_planes))
        # ----->: batch_size * n_points, n_samples, 1, out_planes / share_planes
        w = w.view(-1, n_samples, 1, out_planes // share_planes)

        # v.size: batch_size * n_points, n_samples, share_planes, out_planes / share_planes
        v = v.view(-1, n_samples, share_planes, out_planes // share_planes)

        # outputs.size: batch_size * n_points, n_samples, share_planes, out_planes / share_planes
        # ----------->: batch_size * n_points, share_planes, out_planes / share_planes
        outputs = torch.sum(torch.mul(v, w), dim=1, keepdim=False)
        # ----------->: batch_size * n_points, out_planes
        outputs = outputs.view(-1, out_planes)
        return outputs


class PointTransformerBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        share_planes: int,
        shrink_rate: int,
        radius: int,
    ) -> None:
        super().__init__()

        # Modules 1
        modules = []
        modules.append(nn.Linear(in_planes, out_planes, bias=False))
        modules.append(nn.BatchNorm1d(out_planes))
        modules.append(nn.ReLU(inplace=True))

        self.add_module('modules_1', nn.Sequential(*modules))

        # Modules 2
        modules = []
        modules.append(PointTransformerLayer(
            out_planes,
            out_planes,
            share_planes,
            shrink_rate,
            radius,
        ))
        modules.append(nn.BatchNorm1d(out_planes))
        modules.append(nn.ReLU(inplace=True))

        self.add_module('modules_2', nn.Sequential(*modules))

        # Modules 3
        modules = []
        modules.append(nn.Linear(out_planes, out_planes, bias=False))
        modules.append(nn.BatchNorm1d(out_planes))

        self.add_module('modules_3', nn.Sequential(*modules))

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # points.size: batch_size, n_points, 2
        # features.size: batch_size * n_points, in_planes
        points, features = inputs

        # outputs.size: batch_size * n_points, out_planes
        outputs = self.modules_1(features)
        # ----------->: batch_size * n_points, out_planes
        outputs = self.modules_2((points, outputs))
        # ----------->: batch_size * n_points, out_planes
        outputs = self.modules_3(outputs)

        outputs = F.relu(outputs + features)
        return points, outputs


class PointTransformerEncoderLayer(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        share_planes: int,
        shrink_rate: int,
        radius: int,
        stride: int,
        num_blocks: int,
    ) -> None:
        modules = []
        if stride > 1 and in_planes != out_planes:
            modules.append(TransitionDown(
                in_planes,
                out_planes,
                radius,
                stride,
            ))
            in_planes = out_planes

        for __ in range(num_blocks):
            modules.append(PointTransformerBlock(
                in_planes,
                out_planes,
                share_planes,
                shrink_rate,
                radius,
            ))
            in_planes = out_planes

        super().__init__(*modules)
