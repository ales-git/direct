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
        """Inits :class:`RecurrentVarNetEngine."""
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
        # Generate the output k-space
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_kspace = T.apply_padding(output_kspace, data.get("padding", None))

        # Reconstruct the output image (if needed)
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )

        ########## Custom Code: Save One 3D K-space ################################################################
        # Ensure the folder exists for saving
        output_folder = '/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/pred_3d_kspace/'
        os.makedirs(output_folder, exist_ok=True)

        # Generate a timestamp for uniqueness
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        #kspace_filename = os.path.join(output_folder, f"predicted_3d_kspace_{timestamp}.pt")
        filename = data['filename']
        filename = str(filename).split('/')[-1].split('.')[0]

        kspace_filename = os.path.join(output_folder, filename+".pt")

        # Save the 3D k-space tensor (this will include all coils)
        torch.save(output_kspace, kspace_filename)
        print(f"Saved 3D k-space to: {kspace_filename}")
        ###########################################################################################################

        return output_image, output_kspace
