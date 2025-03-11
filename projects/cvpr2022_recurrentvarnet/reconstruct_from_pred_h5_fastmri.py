import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Set your input and output folders
INPUT_FOLDER = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/output_fastmri_test/pred_h5/"
OUTPUT_FOLDER = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/output_fastmri_test/pred_png/"
NII_OUTPUT_FOLDER = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/output_fastmri_test/pred_nii/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(NII_OUTPUT_FOLDER, exist_ok=True)

def process_h5_file(h5_file):
    """Loads reconstructed images, saves as PNG slices and NIfTI volume."""
    with h5py.File(h5_file, "r") as f:
        reconstruction = f["reconstruction"][:]  # Shape: (slices, height, width)

    base_filename = os.path.splitext(os.path.basename(h5_file))[0]

    # Save each slice as PNG
    for slice_idx in range(reconstruction.shape[0]):
        image = reconstruction[slice_idx]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_slice_{slice_idx}.png")
        plt.imsave(output_path, image, cmap="gray")
        print(f"Saved slice: {output_path}")

    # Save entire volume as NIfTI (.nii.gz)
    nii_img = nib.Nifti1Image(reconstruction, affine=np.eye(4))  # Identity affine for simple orientation
    nii_output_path = os.path.join(NII_OUTPUT_FOLDER, f"{base_filename}.nii.gz")
    nib.save(nii_img, nii_output_path)
    print(f"Saved volume: {nii_output_path}")

def process_all_h5_files():
    """Processes all HDF5 files in the input folder."""
    h5_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".h5")]
    
    if not h5_files:
        print("No HDF5 files found in the input folder.")
        return
    
    for h5_file in h5_files:
        print(f"Processing {h5_file}...")
        process_h5_file(os.path.join(INPUT_FOLDER, h5_file))

# Run the script
if __name__ == "__main__":
    process_all_h5_files()