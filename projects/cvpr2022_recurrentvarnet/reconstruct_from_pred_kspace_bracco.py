import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Set folder paths
kspace_folder = '/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/pred_3d_kspace/'
output_png_folder = '/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/output_bracco_test/output_slices/'
output_nii_folder = '/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/output_bracco_test/output_nifti/'
output_plots_folder = '/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/output_bracco_test/output_plots/'

# Ensure output folders exist
os.makedirs(output_png_folder, exist_ok=True)
os.makedirs(output_nii_folder, exist_ok=True)
os.makedirs(output_plots_folder, exist_ok=True)

# Processing all .pt files in the folder
for file_name in os.listdir(kspace_folder):
    if file_name.endswith('.pt'):
        file_path = os.path.join(kspace_folder, file_name)
        print(f"Processing: {file_name}")
        
        # Load k-space data
        kspace = torch.load(file_path)
        kspace = kspace.squeeze(0)  # Remove coil dimension if necessary
        kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])
        
        # Apply inverse FFT
        pixshift, pixshift2 = 12, -26
        reconstructed_image = np.flip(
            np.moveaxis(
                np.roll(np.roll(abs(np.fft.fftshift(np.fft.fftn(kspace_complex.cpu()))), pixshift, 0), pixshift2, 2), 2, 0),
            axis=0
        )
        
        # Normalize the image
        images = (reconstructed_image - np.min(reconstructed_image)) / (np.max(reconstructed_image) - np.min(reconstructed_image))
        
        # Save slices as PNG
        slice_folder = os.path.join(output_png_folder, file_name.replace('.pt', ''))
        os.makedirs(slice_folder, exist_ok=True)
        
        for i in range(images.shape[0]):  # Axial slices
            plt.imsave(os.path.join(slice_folder, f"slice_{i:03d}.png"), images[i, :, :], cmap='gray', vmin=0, vmax=1)
        
        # Save as .nii.gz file
        nii_file_path = os.path.join(output_nii_folder, file_name.replace('.pt', '.nii.gz'))
        nifti_img = nib.Nifti1Image(reconstructed_image, affine=np.eye(4))
        nib.save(nifti_img, nii_file_path)
        
        # Generate quick visualization plots
        num_slices = 16
        start_idx_axial = (images.shape[0] - num_slices) // 2
        start_idx_coronal = (images.shape[1] - num_slices) // 2
        start_idx_sagittal = (images.shape[2] - num_slices) // 2
        plot_folder = os.path.join(output_plots_folder, file_name.replace('.pt', ''))
        os.makedirs(plot_folder, exist_ok=True)
        
        # Axial view
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(num_slices):
            ax = axes[i // 4, i % 4]
            ax.imshow(images[start_idx_axial + i, :, :], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Axial Slice {start_idx_axial + i}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, "axial_view.png"), dpi=300)
        plt.close()
        
        # Coronal view
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(num_slices):
            ax = axes[i // 4, i % 4]
            ax.imshow(images[:, start_idx_coronal + i, :], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Coronal Slice {start_idx_coronal + i}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, "coronal_view.png"), dpi=300)
        plt.close()
        
        # Sagittal view
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(num_slices):
            ax = axes[i // 4, i % 4]
            ax.imshow(images[:, :, start_idx_sagittal + i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Sagittal Slice {start_idx_sagittal + i}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, "sagittal_view.png"), dpi=300)
        plt.close()
        
        print(f"Saved PNG slices in {slice_folder}, NIfTI volume at {nii_file_path}, and visualization plots in {plot_folder}")
