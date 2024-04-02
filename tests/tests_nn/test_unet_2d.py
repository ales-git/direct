# Copyright (c) DIRECT Contributors

"""Tests for `direct.nn.unet.unet_2d.Unet2d` model."""

import numpy as np
import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.types import InitType
from direct.nn.unet.unet_2d import Unet2d


def create_input(shape):
    data = np.random.randn(*shape).copy()
    data = torch.from_numpy(data).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3, 16, 16],
        [4, 5, 16, 32],
        [3, 4, 32, 32],
        [3, 4, 40, 20],
    ],
)
@pytest.mark.parametrize(
    "num_filters",
    [4, 6, 8],
)
@pytest.mark.parametrize(
    "num_pool_layers",
    [2, 3],
)
@pytest.mark.parametrize(
    "skip",
    [True, False],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
@pytest.mark.parametrize(
    "image_init",
    [InitType.SENSE, InitType.ZERO_FILLED],
)
def test_unet_2d(shape, num_filters, num_pool_layers, skip, normalized, image_init):
    model = Unet2d(
        fft2,
        ifft2,
        num_filters=num_filters,
        num_pool_layers=num_pool_layers,
        skip_connection=skip,
        normalized=normalized,
        image_initialization=image_init,
        dropout_probability=0.05,
    ).cpu()

    data = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()

    out = model(data, sens)

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
