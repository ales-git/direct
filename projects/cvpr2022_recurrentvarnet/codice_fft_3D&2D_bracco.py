import h5py
import numpy as np
import matplotlib.pyplot as plt

input_path = '/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5/20240722_133608_IMG_08_24_l01_32dpost_1_52_13_vHD.h5'


# Load the HDF5 file
with h5py.File(input_path, "r") as f:
    kspace = f['kspace'][()]
    pixshift = int(f['pixshift'][0])
    pixshift2 = int(f['pixshift'][1])
    reference_image = f['image'][()]

# Convert to complex tensor if needed
kspace_complex = np.asarray(kspace)


# Apply the transformation to get the reconstructed image (full FFT)
reconstructed_image = np.flip(
    np.moveaxis(
        np.roll(np.roll(
            np.abs(np.fft.fftshift(
                np.fft.fftn(
                    np.rot90(np.flip(kspace_complex, 1), -1)
                ))), 
            pixshift, axis=0), 
            pixshift2, axis=2), 
        2, 0),
    axis=0
)


# Normalize both images for comparison
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

reconstructed_image_norm = normalize(reconstructed_image)
reference_image_norm = normalize(reference_image)

# Compute the difference
diff = np.abs(reconstructed_image_norm - reference_image_norm)

# Show a middle slice
mid_slice = reconstructed_image_norm.shape[0] // 2

#### 2d case
# Reorient input to match the 3D case
kspace_reoriented = np.rot90(np.flip(kspace_complex, 1), -1)

# 1D FFT along axis 2 (last axis)
partial_fft = np.fft.fft(kspace_reoriented, axis=2)

# Apply 2D FFT slice-by-slice (same axis order)
test_ft_2d = np.zeros_like(partial_fft, dtype=np.complex64)
for i in range(partial_fft.shape[2]):
    test_ft_2d[:, :, i] = np.fft.fft2(partial_fft[:, :, i])

# Apply fftshift and magnitude (NOT ifftn)
reconstructed_image_alt = np.abs(np.fft.fftshift(test_ft_2d))

# Apply the same shifts and axis changes as in the 3D version
reconstructed_image_alt = np.flip(
    np.moveaxis(
        np.roll(np.roll(reconstructed_image_alt, pixshift, axis=0),
                pixshift2, axis=2),
        2, 0),
    axis=0
)


# Normalize
reconstructed_image_alt_norm = normalize(reconstructed_image_alt)

# Compute diff from reference
diff_alt = np.abs(reconstructed_image_alt_norm - reference_image_norm)

# Show all four plots
plt.figure(figsize=(16, 4))

# Plot the reconstructed image using full FFT
plt.subplot(1, 4, 1)
plt.title("Reconstructed (Full FFTN)")
plt.imshow(reconstructed_image_norm[mid_slice], cmap='gray')

# Plot the reconstructed image using 1D FFT followed by 2D FFT
plt.subplot(1, 4, 2)
plt.title("Reconstructed (1D FFT → 2D FFT)")
plt.imshow(reconstructed_image_alt_norm[mid_slice], cmap='gray')

# Plot the reference image
plt.subplot(1, 4, 3)
plt.title("Reference Image")
plt.imshow(reference_image_norm[mid_slice], cmap='gray')

# Plot the difference from the alternate reconstruction
plt.subplot(1, 4, 4)
plt.title("Alt Diff from Reference")
plt.imshow(diff_alt[mid_slice], cmap='hot')
plt.colorbar()

plt.tight_layout()
plt.show()
