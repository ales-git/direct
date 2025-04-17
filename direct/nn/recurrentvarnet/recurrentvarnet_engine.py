# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import nn
import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np


class RecurrentVarNetEngine(MRIModelEngine):
    """Recurrent Variational Network Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Initialize the RecurrentVarNetEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        ############ Custom Code: Save undersampled (masked) 3D k-space ############
        output_folder_masked = '/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/masked_3d_kspace_2D/'
        os.makedirs(output_folder_masked, exist_ok=True)
        filename = data['filename']
        filename = str(filename).split('/')[-1].split('.')[0]

        undersampled_kspace = data["masked_kspace"]
        print(f"Undersampled k-space shape: {undersampled_kspace.shape}")  # Debug print
        torch.save(undersampled_kspace, os.path.join(output_folder_masked, filename + ".pt"))
        print(f"Saved undersampled k-space to: {filename}.pt")
        ############################################################################

        # Generate the output k-space from the network
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"]
        )
        output_kspace = T.apply_padding(output_kspace, data.get("padding", None))

        # Reconstruct output image (if needed)
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )

        ############ Custom Code: Save each 2D predicted k-space slice ############
        output_folder_slices = '/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/pred_3d_kspace_2D/'
        output_folder_kspace_pred = '/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/pred_3d_kspace/'
        os.makedirs(output_folder_slices, exist_ok=True)
        os.makedirs(output_folder_kspace_pred, exist_ok=True)

        # Assume output_kspace shape: [1, num_slices, height, width, 2]
        predicted_volume = output_kspace[0]  # Remove batch dim → shape [num_slices, 128, 128, 2]
        num_slices = predicted_volume.shape[0]
        print(f"Number of slices: {num_slices}")  # Debug print

        print('##############################')
        print("masked_kspace shape:", data["masked_kspace"].shape)
        print("sampling_mask shape:", data["sampling_mask"].shape)
        print("output_kspace shape:", output_kspace.shape)
        print("predicted_volume shape:", predicted_volume.shape)

        kspace3d_filename = os.path.join(output_folder_kspace_pred, f"{filename}.pt")
        torch.save(predicted_volume,kspace3d_filename)
        print(f"Saved 3d kspace for {kspace3d_filename}")

        for idx in range(num_slices):
            slice_tensor = predicted_volume[idx]  # Shape: [128, 128, 2]
            print(f"Saving slice {idx} with shape: {slice_tensor.shape}")  # Debug print
            slice_filename = os.path.join(output_folder_slices, f"{filename}_slice_{idx:03d}.pt")
            torch.save(slice_tensor, slice_filename)
            print(f"Saved slice {idx} as {slice_filename}")
        ############################################################################

        ############ Custom Code: Save each 2D sampling mask slice ############
        output_folder_mask = '/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/sampling_mask_2D/'
        os.makedirs(output_folder_mask, exist_ok=True)

  
        if "sampling_mask" in data:
            # Extract the 2D mask by squeezing unnecessary dimensions
            # [1, 1, 128, 128, 1] -> [128, 128]
            sampling_mask = data["sampling_mask"].squeeze()  # Removing all singleton dimensions
            
            print(f"Saving sampling mask with shape: {sampling_mask.shape}")  # Debug print
            
            # Now save the mask as a 2D tensor
            mask_filename = os.path.join(output_folder_mask, f"{filename}_mask_2d.pt")
            
            # Save as tensor
            torch.save(sampling_mask, mask_filename)
            print(f"Saved sampling mask as {mask_filename}")


        ############################################################################

        return output_image, output_kspace
