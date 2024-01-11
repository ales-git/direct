# Copyright (c) DIRECT Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from direct.config.defaults import ModelConfig
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.types import ActivationType


@dataclass
class PolicyConfig(ModelConfig):
    acceleration: float = MISSING
    center_fraction: float = MISSING
    sampling_dimension: PolicySamplingDimension = MISSING
    st_slope: float = 10
    st_clamp: bool = False
    use_softplus: bool = True
    slope: float = 10
    fix_sign_leakage: bool = True


@dataclass
class ParameterizedPolicyConfig(PolicyConfig):
    kspace_shape: tuple[int, ...] = MISSING


@dataclass
class Parameterized2dPolicyConfig(ParameterizedPolicyConfig):
    pass


@dataclass
class Parameterized3dPolicyConfig(ParameterizedPolicyConfig):
    pass


@dataclass
class ParameterizedDynamic2dPolicyConfig(ParameterizedPolicyConfig):
    num_time_steps: int = MISSING


@dataclass
class ParameterizedMultislice2dPolicyConfig(ParameterizedPolicyConfig):
    num_slices: int = MISSING


@dataclass
class StraightThroughPolicyConfig(PolicyConfig):
    kspace_shape: tuple[int, int] = MISSING
    num_layers: int = 2
    num_fc_layers: int = 3
    fc_size: int = 256
    kspace_sampler: bool = False
    sampler_detach_mask: bool = False
    drop_prob: float = 0.0
    activation: ActivationType = ActivationType.LEAKYRELU
    sampling_type: PolicySamplingType = PolicySamplingType.STATIC
    num_time_steps: Optional[int] = None
    num_slices: Optional[int] = None
