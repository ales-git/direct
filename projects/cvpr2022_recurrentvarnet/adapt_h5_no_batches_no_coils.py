import h5py
import numpy as np
import xml.etree.ElementTree as ET
import os

# --- SETTINGS ---
INPUT_H5_DIR = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5/"
OUTPUT_H5_DIR = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5_adapted/"
ORIGINAL_SHAPE = (256, 256)  # Original k-space shape

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_H5_DIR, exist_ok=True)

# --- FUNCTION TO CREATE ISMRMRD HEADER ---
def generate_ismrmrd_header(kspace_shape):
    root = ET.Element("ismrmrdHeader", xmlns="http://www.ismrm.org/ISMRMRD")
    encoding = ET.SubElement(root, "encoding")

    # Encoded Space (Original k-space size)
    encoded_space = ET.SubElement(encoding, "encodedSpace")
    matrix_size = ET.SubElement(encoded_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = str(kspace_shape[3])  # Width
    ET.SubElement(matrix_size, "y").text = str(kspace_shape[2])  # Height
    ET.SubElement(matrix_size, "z").text = str(kspace_shape[1])  # Number of slices

    # Recon Space (Final reconstructed image size)
    recon_space = ET.SubElement(encoding, "reconSpace")
    matrix_size = ET.SubElement(recon_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = str(kspace_shape[3])  
    ET.SubElement(matrix_size, "y").text = str(kspace_shape[2])  
    ET.SubElement(matrix_size, "z").text = str(kspace_shape[1])  

    # Encoding Limits
    encoding_limits = ET.SubElement(encoding, "encodingLimits")
    kspace_encoding_step_1 = ET.SubElement(encoding_limits, "kspace_encoding_step_1")
    ET.SubElement(kspace_encoding_step_1, "minimum").text = "0"
    ET.SubElement(kspace_encoding_step_1, "maximum").text = str(kspace_shape[2] - 1)
    ET.SubElement(kspace_encoding_step_1, "center").text = str(kspace_shape[2] // 2)

    kspace_encoding_step_2 = ET.SubElement(encoding_limits, "kspace_encoding_step_2")
    ET.SubElement(kspace_encoding_step_2, "minimum").text = "0"
    ET.SubElement(kspace_encoding_step_2, "maximum").text = "0"
    ET.SubElement(kspace_encoding_step_2, "center").text = "0"

    return ET.tostring(root, encoding="utf-8").decode()

# --- CONVERTING EACH HDF5 FILE ---
for file_name in os.listdir(INPUT_H5_DIR):
    if not file_name.endswith(".h5"):
        continue  # Skip non-HDF5 files

    input_path = os.path.join(INPUT_H5_DIR, file_name)
    output_path = os.path.join(OUTPUT_H5_DIR, file_name)

    with h5py.File(input_path, "r") as f_in:
        # Load k-space from input file
        kspace = f_in["dataset_name"][:]  # Adjust key if needed

        # Ensure k-space is in correct format (originally single coil)
        kspace = np.expand_dims(kspace, axis=0)  # Add coil dimension (set to 1)

        # Keep the entire 256x256x256 volume (no coil duplication)
        kspace = kspace.astype(np.complex128)  # Convert to complex128

        # Generate metadata (ismrmrd_header)
        ismrmrd_header = generate_ismrmrd_header(kspace.shape)

        # Generate a dummy sensitivity map (all ones, single coil)
        sensitivity_map = np.ones_like(kspace, dtype=np.complex128)

        # Create output file name
        output_file_name = f"{os.path.splitext(file_name)[0]}_full_volume.h5"
        output_file_path = os.path.join(OUTPUT_H5_DIR, output_file_name)

        # Save to HDF5 file
        with h5py.File(output_file_path, "w") as f_out:
            f_out.create_dataset("ismrmrd_header", data=np.bytes_(ismrmrd_header))
            f_out.create_dataset("kspace", data=kspace)
            f_out.create_dataset("sensitivity_map", data=sensitivity_map)  

            # Generate and store a dummy mask (all ones)
            #mask = np.ones(kspace.shape[-2], dtype=np.float32)  # Only store 1D mask
            #f_out.create_dataset("mask", data=mask)

        print(f"Processed: {output_file_name}")

print("\nAll files converted and saved to:", OUTPUT_H5_DIR)
