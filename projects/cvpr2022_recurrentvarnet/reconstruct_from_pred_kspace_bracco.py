import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from collections import defaultdict

# Paths
kspace_folder = '/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/pred_3d_kspace_2D/'
output_base = '/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/output_bracco_test_2d/'
output_png = os.path.join(output_base, 'output_slices')
output_nii = os.path.join(output_base, 'output_nifti')
output_plots = os.path.join(output_base, 'output_plots')
os.makedirs(output_png, exist_ok=True)
os.makedirs(output_nii, exist_ok=True)
os.makedirs(output_plots, exist_ok=True)

# Normalize helper
def normalize(img):
    img = np.abs(img)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

# Parse and group by volume name
volume_dict = defaultdict(list)
pattern = re.compile(r"(.+)_slice_(\d+)\.pt")

for fname in sorted(os.listdir(kspace_folder)):
    if not fname.endswith('.pt'):
        continue
    match = pattern.match(fname)
    if match:
        volname, slice_idx = match.group(1), int(match.group(2))
        volume_dict[volname].append((slice_idx, fname))

# Process each volume
for volname, slices in volume_dict.items():
    print(f"Reconstructing volume: {volname}")
    slices_sorted = sorted(slices, key=lambda x: x[0])
    volume_slices = []

    for slice_idx, fname in slices_sorted:
        fpath = os.path.join(kspace_folder, fname)
        kspace = torch.load(fpath).cpu()

        # Check shape and convert to complex numpy
        if kspace.ndim == 4:
            kspace = kspace.squeeze(0)  # [128, 128, 2]
        if kspace.shape[-1] != 2:
            raise ValueError(f"Unexpected last dimension in k-space: {kspace.shape}")
        
        # Construct complex k-space array (real + imag)
        kspace_complex = kspace[..., 0] + 1j * kspace[..., 1]  # shape [128, 128]

        # Debugging step: Check the shape before FFT
        print(f"Shape of kspace_complex (slice {slice_idx}): {kspace_complex.shape}")

        # 2D iFFT
        img = np.fft.ifft2(kspace_complex)
        img = np.fft.fftshift(img)
        img_abs = np.abs(img)
        volume_slices.append(img_abs)

    volume_np = np.stack(volume_slices, axis=0)  # shape [D, H, W]
    volume_np_norm = normalize(volume_np)

    # Save PNG slices
    slice_out_dir = os.path.join(output_png, volname)
    os.makedirs(slice_out_dir, exist_ok=True)
    for i, slice_img in enumerate(volume_np_norm):
        plt.imsave(os.path.join(slice_out_dir, f"slice_{i:03d}.png"), slice_img, cmap='gray', vmin=0, vmax=1)

    # Save NIfTI
    nii_path = os.path.join(output_nii, f"{volname}.nii.gz")
    nib.save(nib.Nifti1Image(volume_np.astype(np.float32), affine=np.eye(4)), nii_path)

    # Save axial/coronal/sagittal views
    plot_out_dir = os.path.join(output_plots, volname)
    os.makedirs(plot_out_dir, exist_ok=True)
    num_slices = 16

    def plot_views(volume, axis, title):
        start_idx = (volume.shape[axis] - num_slices) // 2
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(num_slices):
            idx = start_idx + i
            ax = axs[i // 4, i % 4]
            if axis == 0:
                ax.imshow(volume[idx, :, :], cmap='gray', vmin=0, vmax=1)
            elif axis == 1:
                ax.imshow(volume[:, idx, :], cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(volume[:, :, idx], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_out_dir, f"{title.lower()}_view.png"), dpi=300)
        plt.close()

    plot_views(volume_np_norm, axis=0, title="Axial")
    plot_views(volume_np_norm, axis=1, title="Coronal")
    plot_views(volume_np_norm, axis=2, title="Sagittal")

    print(f"Saved volume {volname}: PNGs → {slice_out_dir}, NIfTI → {nii_path}, Views → {plot_out_dir}")
