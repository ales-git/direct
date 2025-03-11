import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Set your input folder
INPUT_FOLDER = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5_adapted/"


def process_h5_file(h5_file):
    """Loads k-space data, performs inverse Fourier transform, and visualizes slices in different directions."""
    with h5py.File(h5_file, "r") as f:
        kspace = f["kspace"][:]  # Shape: (coils, slices, height, width)

    base_filename = os.path.splitext(os.path.basename(h5_file))[0]
    kspace = kspace[0,:,:,:]

    ''' 
    # Perform inverse Fourier transform to get image space data
    pixshift = 12
    pixshift2 = -26
    images = np.flip(np.moveaxis(np.roll(np.roll(abs(np.fft.fftshift(np.fft.fftn(kspace))), pixshift,0) , pixshift2, 2) ,2,0), axis = 0)

    #images = np.abs(images)  # Take magnitude for visualization

    # Print the shape of the images array for debugging
    print(f"Shape of images array: {images.shape}")

    # Normalize the images (optional, but helps with visualization)
    images = (images - np.min(images)) / (np.max(images) - np.min(images))  

    # Define grid size (4x4)
    rows, cols = 4, 4
    num_slices = rows * cols  # 16 slices

    # Calculate the start index for the central slices in each direction
    start_idx_axial = (images.shape[0] - num_slices) // 2
    start_idx_coronal = (images.shape[1] - num_slices) // 2
    start_idx_sagittal = (images.shape[2] - num_slices) // 2

    # Plot the central 16 slices in axial direction
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(num_slices):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[start_idx_axial + i, :, :], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Axial Slice {start_idx_axial + i}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{base_filename}_images_axial.png", dpi=300)

    # Plot the central 16 slices in coronal direction
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(num_slices):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[:, start_idx_coronal + i, :], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Coronal Slice {start_idx_coronal + i}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{base_filename}_images_coronal.png", dpi=300)

    # Plot the central 16 slices in sagittal direction
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(num_slices):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[:, :, start_idx_sagittal + i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Sagittal Slice {start_idx_sagittal + i}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{base_filename}_images_sagittal.png", dpi=300)

    print(f"Central slices plotted and saved in axial, coronal, and sagittal directions for {base_filename}.")'
    '''
    return kspace, base_filename

def process_all_h5_files():
    """Processes all HDF5 files in the input folder."""
    h5_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".h5")]
    kspace_list = []
    
    if not h5_files:
        print("No HDF5 files found in the input folder.")
        return
    
    for h5_file in h5_files:
        print(f"Processing {h5_file}...")
        kspace, base_filename = process_h5_file(os.path.join(INPUT_FOLDER, h5_file))
        kspace_list.append(kspace)

    return kspace_list, base_filename

def process_all_kspaces(kspace_list, base_filename):
    kspace = np.concatenate(kspace_list, axis=0)
    # Perform inverse Fourier transform to get image space data
    pixshift = 12
    pixshift2 = -26
    images = np.flip(np.moveaxis(np.roll(np.roll(abs(np.fft.fftshift(np.fft.fftn(kspace))), pixshift,0) , pixshift2, 2) ,2,0), axis = 0)

    # Print the shape of the images array for debugging
    print(f"Shape of images array: {images.shape}")

    # Normalize the images (optional, but helps with visualization)
    images = (images - np.min(images)) / (np.max(images) - np.min(images))  

    # Define grid size (4x4)
    rows, cols = 4, 4
    num_slices = rows * cols  # 16 slices

    # Calculate the start index for the central slices in each direction
    start_idx_axial = (images.shape[0] - num_slices) // 2
    start_idx_coronal = (images.shape[1] - num_slices) // 2
    start_idx_sagittal = (images.shape[2] - num_slices) // 2

    # Plot the central 16 slices in axial direction
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(num_slices):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[start_idx_axial + i, :, :], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Axial Slice {start_idx_axial + i}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{base_filename}_images_axial_complete.png", dpi=300)

    # Plot the central 16 slices in coronal direction
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(num_slices):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[:, start_idx_coronal + i, :], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Coronal Slice {start_idx_coronal + i}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{base_filename}_images_coronal_complete.png", dpi=300)

    # Plot the central 16 slices in sagittal direction
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(num_slices):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[:, :, start_idx_sagittal + i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Sagittal Slice {start_idx_sagittal + i}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{base_filename}_images_sagittal_complete.png", dpi=300)

    print(f"Central slices plotted and saved in axial, coronal, and sagittal directions for {base_filename}.")

# Run the script
if __name__ == "__main__":
    kspace_list, base_filename = process_all_h5_files()
    process_all_kspaces(kspace_list, base_filename)