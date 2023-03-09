# -*- coding: utf-8 -*-

import logging
import torch

from torch import Tensor


_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


def circular_grouping_index(x: Tensor, r: int) -> Tensor:
    batch_size, n_points, n_planes = x.size()

    # indexes.size: batch_size, n_points, n_samples
    indexes = torch.tensor([[[
        k * n_points + (i + s) % n_points
        for s in range(-r, r + 1) if s != 0
    ] for i in range(n_points)] for k in range(batch_size)])

    return indexes
