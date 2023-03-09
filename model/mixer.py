# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops.layers.torch import Reduce
from torch import Tensor
from typing import Tuple

from torch_scatter import scatter_softmax
from torch_scatter import scatter_sum

from .utils import circular_grouping_index


_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


class SymmetricTransitionDown(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        radius: int,
        stride: int,
    ) -> None:
        super().__init__()

        n_samples = radius * 2

        self.radius = radius
        self.stride = stride

        # Modules 1
        modules = []
        modules.append(nn.Linear(2 + in_planes, in_planes, bias=False))
        modules.append(nn.BatchNorm1d(in_planes))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(in_planes, 1))

        self.add_module('modules_1', nn.Sequential(*modules))

        # Modules 2
        modules = []
        modules.append(nn.Linear(in_planes, out_planes, bias=False))
        modules.append(nn.BatchNorm1d(out_planes))
        modules.append(nn.ReLU(inplace=True))

        self.add_module('modules_2', nn.Sequential(*modules))

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

        # w.size: batch_size * (n_points / stride) * n_samples, 1
        w = self.modules_1(torch.cat((translation, nn_features), dim=1))
        # ----->: batch_size * n_points / stride, n_samples, 1
        w = F.softmax(w.view(-1, n_samples, 1), dim=1)

        # y.size: batch_size * n_points * n_samples, out_planes
        y = self.modules_2(features[indexes])
        # ----->: batch_size, n_points, n_samples, out_planes
        y = y.view(batch_size, n_points, n_samples, -1)
        # ----->: batch_size, n_points / stride, n_samples, out_planes
        y = y[:, ::self.stride]

        # y.size: batch_size * n_points / stride, n_samples, out_planes
        __, __, __, out_planes = y.size()
        y = torch.reshape(y, (-1, n_samples, out_planes))

        # outputs.size: batch_size * n_points / stride, out_planes
        outputs = torch.sum(torch.mul(y, w), dim=1, keepdim=False)

        # points.size: batch_size, n_points, 2
        # ---------->: batch_size, n_points / stride, 2
        points = points[:, ::self.stride]

        return points, outputs


class BilinearFeedForward(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
    ) -> None:
        super().__init__()

        self.add_module('f', nn.Bilinear(in_planes, in_planes, out_planes))

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: batch_size, n_points, in_planes
        outputs = self.f(inputs, inputs)
        return outputs


class IntraSetLayer(nn.Module):
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
        n_samples = radius * 2
        assert out_planes % n_samples == 0

        self.share_planes = share_planes
        self.radius = radius

        # Modules 1
        modules = []
        modules.append(nn.Linear(2 + in_planes, n_samples))
        modules.append(nn.ReLU(inplace=True))
        modules.append(BilinearFeedForward(n_samples, n_samples))

        self.add_module('modules_1', nn.Sequential(*modules))

        # Position embedding
        modules = []
        modules.append(nn.Linear(2, 2, bias=False))
        modules.append(nn.BatchNorm1d(2))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(2, out_planes))

        self.add_module('linear_p', nn.Sequential(*modules))

        modules = []
        modules.append(Rearrange('n k (a b) -> n k a b', b=n_samples))
        modules.append(Reduce('n k a b -> (n k) b', 'sum', b=n_samples))

        self.add_module('shrink_p', nn.Sequential(*modules))

        # Modules 2
        modules = []

        in_features = n_samples * 2
        out_features = mid_planes
        modules.append(nn.Linear(in_features, out_features, bias=False))
        modules.append(nn.BatchNorm1d(out_features))
        modules.append(nn.ReLU(inplace=True))

        in_features = out_features
        out_features = mid_planes // share_planes
        modules.append(nn.Linear(in_features, out_features, bias=False))
        modules.append(nn.BatchNorm1d(out_features))
        modules.append(nn.ReLU(inplace=True))

        in_features = out_features
        out_features = out_planes // share_planes
        modules.append(nn.Linear(in_features, out_features))

        modules.append(Rearrange('(n k) c -> n k c', k=n_samples))
        modules.append(nn.Softmax(dim=1))

        self.add_module('modules_2', nn.Sequential(*modules))

        # Modules 3
        modules = []
        modules.append(nn.Linear(in_planes, out_planes))

        self.add_module('modules_3', nn.Sequential(*modules))

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
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

        # nn_features.size: batch_size * n_points * n_samples, in_planes
        nn_features = features[indexes]

        # energy.size: batch_size * n_points * n_samples, n_samples
        energy = self.modules_1(torch.cat((translation, nn_features), dim=1))

        # p_embed.size: batch_size * n_points * n_samples, out_planes
        p_embed = self.linear_p(translation)
        # ----------->: batch_size * n_points, n_samples, out_planes
        __, out_planes = p_embed.size()
        p_embed = p_embed.view(-1, n_samples, out_planes)

        # p_embed_shrink.size: batch_size * n_points * n_samples, n_samples
        p_embed_shrink = self.shrink_p(p_embed)

        # w.size: batch_size * n_points, n_samples, out_planes / share_planes
        w = self.modules_2(torch.cat((energy, p_embed_shrink), dim=1))
        # w.size: batch_size * n_points, n_samples, 1, out_planes / share_planes
        w = torch.unsqueeze(w, 2)

        # v.size: batch_size * n_points, out_planes
        v = self.modules_3(features)
        # ----->: batch_size * n_points, out_planes
        v = v.view(-1, out_planes)
        # ----->: batch_size * n_points * n_samples, out_planes
        v = v[indexes]
        # ----->: batch_size * n_points, n_samples, out_planes
        v = v.view(-1, n_samples, out_planes)

        # a.size: batch_size * n_points, n_samples, out_planes
        a = v + p_embed
        # ----->: batch_size * n_points, n_samples, share_planes, out_planes / share_planes
        share_planes = self.share_planes
        a = a.view(-1, n_samples, share_planes, out_planes // share_planes)

        # outputs.size: batch_size * n_points, n_samples, out_planes
        outputs = torch.reshape(torch.mul(a, w), (-1, n_samples, out_planes))

        return outputs, translation, indexes


class InterSetLayer(nn.Module):
    def __init__(
        self,
        out_planes: int,
        share_planes: int,
    ) -> None:
        super().__init__()

        mid_planes = out_planes // share_planes

        self.share_planes = share_planes

        # Modules
        modules = []
        modules.append(nn.Linear(out_planes, mid_planes))

        self.add_module('modules', nn.Sequential(*modules))

        # Modules
        modules = []
        modules.append(nn.Linear(out_planes, mid_planes))

        self.add_module('modules_x', nn.Sequential(*modules))

        # Modules
        modules = []
        modules.append(nn.Linear(2, 2, bias=False))
        modules.append(nn.BatchNorm1d(2))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(2, out_planes))

        self.add_module('modules_p', nn.Sequential(*modules))

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        # outputs.size: batch_size * n_points, n_samples, out_planes
        # translation.size: batch_size * n_points, n_samples, 2
        # indexes.size: batch_size * n_points * n_samples
        outputs, translation, indexes = inputs

        size, n_samples, out_planes = outputs.size()

        # indexes.size: batch_size * n_points * n_samples, 1
        indexes = torch.unsqueeze(indexes, 1)

        # features.size: batch_size * n_points, out_planes
        features = torch.sum(outputs, dim=1, keepdim=False)

        # outputs.size: batch_size * n_points * n_samples, out_planes
        outputs = outputs.view(-1, out_planes)

        # translation.size: batch_size * n_points * n_samples, 2
        # p_embed.size: batch_size * n_points * n_samples, out_planes
        p_embed = self.modules_p(translation.view(-1, 2))

        # x.size: batch_size * n_points * n_samples, mid_planes
        x = self.modules(outputs + p_embed)
        # w.size: batch_size * n_points * n_samples, mid_planes
        w = scatter_softmax(x, indexes, dim=0)
        # v.size: batch_size * n_points * n_samples, mid_planes
        v = self.modules_x(outputs)

        # residual.size: batch_size * n_points, mid_planes
        residual = scatter_sum(torch.mul(v, w), indexes, dim=0, dim_size=size)

        # outputs.size: batch_size * n_points, out_planes
        outputs = features + residual.repeat(1, self.share_planes)
        return outputs


class PointMixerBlock(nn.Module):
    def __init__(
        self,
        n_planes: int,
        share_planes: int,
        shrink_rate: int,
        radius: int,
    ) -> None:
        super().__init__()

        # Modules 1
        modules = []
        modules.append(nn.Linear(n_planes, n_planes, bias=False))
        modules.append(nn.BatchNorm1d(n_planes))
        modules.append(nn.ReLU(inplace=True))

        self.add_module('modules_1', nn.Sequential(*modules))

        # Modules 2
        modules = []
        modules.append(IntraSetLayer(
            n_planes,
            n_planes,
            share_planes,
            shrink_rate,
            radius,
        ))
        modules.append(InterSetLayer(
            n_planes,
            share_planes,
        ))

        self.add_module('modules_2', nn.Sequential(*modules))

        # Modules 1
        modules = []
        modules.append(nn.Linear(n_planes, n_planes, bias=False))
        modules.append(nn.BatchNorm1d(n_planes))

        self.add_module('modules_3', nn.Sequential(*modules))

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # points.size: batch_size, n_points, 2
        # features.size: batch_size * n_points, n_planes
        points, features = inputs

        # outputs.size: batch_size * n_points, n_planes
        outputs = self.modules_1(features)
        # ----------->: batch_size * n_points, n_planes
        outputs = self.modules_2((points, outputs))
        # ----------->: batch_size * n_points, n_planes
        outputs = self.modules_3(outputs)

        outputs = F.relu(outputs + features)
        return points, outputs


class PointMixerEncoderLayer(nn.Sequential):
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
            modules.append(SymmetricTransitionDown(
                in_planes,
                out_planes,
                radius,
                stride,
            ))
            in_planes = out_planes

        for __ in range(num_blocks):
            modules.append(PointMixerBlock(
                in_planes,
                share_planes,
                shrink_rate,
                radius,
            ))

        super().__init__(*modules)
