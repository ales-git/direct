import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- SETTINGS ---
original_h5_file = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5/20240722_133608_IMG_08_24_l01_32dpost_1_52_13_vHD.h5"
adapted_h5_file = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5_adapted_2D/20240722_133608_IMG_08_24_l01_32dpost_1_52_13_vHD.h5"

# --- LOAD THE ORIGINAL K-SPACE ---
with h5py.File(original_h5_file, 'r') as f_in:
    kspace_original = f_in["kspace"][:]  # Assuming the shape is (width, height, slices)

# --- LOAD THE ADAPTED 2D K-SPACE ---
with h5py.File(adapted_h5_file, 'r') as f_in:
    kspace_2d = f_in["kspace"][:]  # Assuming the shape is (width, height, slices)

# --- Select the slice to compare (for example, slice 64 in the middle) ---
slice_idx = 64

# Extract slices from both the original and adapted k-space for comparison
kspace_original_slice = np.abs(kspace_original[:, :, slice_idx])  # Take the absolute value for visualization

# Check for unnecessary singleton dimension in adapted 2D k-space
kspace_2d_slice = np.abs(kspace_2d[:, :, slice_idx])  # Take the absolute value for visualization
if kspace_2d_slice.ndim == 3:
    kspace_2d_slice = np.squeeze(kspace_2d_slice)  # Remove any singleton dimension

# --- Plotting side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(kspace_original_slice, cmap='gray')
axes[0].set_title('Original K-Space Slice')
axes[0].axis('off')

axes[1].imshow(kspace_2d_slice, cmap='gray')
axes[1].set_title('Adapted 2D K-Space Slice')
axes[1].axis('off')

plt.tight_layout()
plt.show()
