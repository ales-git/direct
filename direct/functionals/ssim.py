# coding=utf-8
# Copyright (c) DIRECT Contributors

# Taken from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
# Licensed under MIT.
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
# Some changes are made to work together with DIRECT.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SSIMLoss", "SSIM3DLoss")


class SSIMLoss(nn.Module):
    """SSIM loss module.

    From: https://github.com/facebookresearch/fastMRI/blob/master/fastmri/losses.py
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Parameters
        ----------
        win_size: int
            Window size for SSIM calculation. Default: 7.
        k1: float
            k1 parameter for SSIM calculation. Default: 0.1.
        k2: float
            k2 parameter for SSIM calculation. Default: 0.03.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class SSIM3DLoss(nn.Module):
    """SSIM loss module for 3D data."""

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Parameters
        ----------
        win_size: int
            Window size for SSIM calculation. Default: 7.
        k1: float
            k1 parameter for SSIM calculation. Default: 0.1.
        k2: float
            k2 parameter for SSIM calculation. Default: 0.03.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SSIM3Dloss`.

        Parameters
        ----------
        X : torch.Tensor
        Y : torch.Tensor
        data_range : torch.Tensor
        """
        data_range = data_range[:, None, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        # window size across last dimension is chosen to be the last dimension size if smaller than given window size
        win_size_z = min(self.win_size, X.size(2))

        NP = win_size_z * self.win_size**2
        w = torch.ones(1, 1, win_size_z, self.win_size, self.win_size, device=X.device) / NP
        cov_norm = NP / (NP - 1)

        ux = F.conv3d(X, w)
        uy = F.conv3d(Y, w)
        uxx = F.conv3d(X * X, w)
        uyy = F.conv3d(Y * Y, w)
        uxy = F.conv3d(X * Y, w)

        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
