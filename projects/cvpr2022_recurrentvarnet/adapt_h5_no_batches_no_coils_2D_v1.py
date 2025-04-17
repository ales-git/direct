import h5py
import numpy as np
import os
import xml.etree.ElementTree as ET

# --- SETTINGS ---
INPUT_H5_DIR = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5/"
OUTPUT_H5_DIR = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5_adapted_2D/"
ORIGINAL_SHAPE = (128, 128)

os.makedirs(OUTPUT_H5_DIR, exist_ok=True)

# --- FUNCTION TO CREATE ISMRMRD HEADER ---
def generate_ismrmrd_header(kspace_shape):
    root = ET.Element("ismrmrdHeader", xmlns="http://www.ismrm.org/ISMRMRD")
    encoding = ET.SubElement(root, "encoding")

    encoded_space = ET.SubElement(encoding, "encodedSpace")
    matrix_size = ET.SubElement(encoded_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = str(kspace_shape[3])  # Width
    ET.SubElement(matrix_size, "y").text = str(kspace_shape[2])  # Height
    ET.SubElement(matrix_size, "z").text = str(kspace_shape[1])  # Number of slices

    recon_space = ET.SubElement(encoding, "reconSpace")
    matrix_size = ET.SubElement(recon_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = str(kspace_shape[3])
    ET.SubElement(matrix_size, "y").text = str(kspace_shape[2])
    ET.SubElement(matrix_size, "z").text = str(kspace_shape[1])

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
        continue

    input_path = os.path.join(INPUT_H5_DIR, file_name)
    output_path = os.path.join(OUTPUT_H5_DIR, file_name)

    with h5py.File(input_path, "r") as f_in:
        kspace = f_in["kspace"][:]

        # Ensure shape is (1, 128, 128, 128) [coil, z, y, x]
        kspace = np.expand_dims(kspace, axis=0).astype(np.complex128)
        print(f"Original kspace shape: {kspace.shape}")

        # --- Apply partial FFT along z-axis (slices) ---
        partial_fft = np.flip(np.roll(np.fft.fft(kspace, axis=2), 64, 2),2)
        print(f"Partial FFT along z-axis, shape: {partial_fft.shape}")

        # --- Apply 2D FFT slice-by-slice (along the last two axes) ---
        #kspace_2d_slices = np.zeros_like(partial_fft, dtype=np.complex128)
        #for i in range(partial_fft.shape[2]):  # Iterating over slices (axis=2)
        #    kspace_2d_slices[:, :, i] = np.fft.fft2(partial_fft[:, :, i])
        #print(f"Shape after 2D FFT along slices: {kspace_2d_slices.shape}")
        kspace_2d_slices = partial_fft

        # Generate ISMRMRD header
        ismrmrd_header = generate_ismrmrd_header(kspace_2d_slices.shape)
        sensitivity_map = np.ones_like(kspace_2d_slices, dtype=np.complex128)

        output_file_name = f"{os.path.splitext(file_name)[0]}.h5"
        output_file_path = os.path.join(OUTPUT_H5_DIR, output_file_name)

        with h5py.File(output_file_path, "w") as f_out:
            f_out.create_dataset("ismrmrd_header", data=np.bytes_(ismrmrd_header))
            f_out.create_dataset("kspace", data=kspace_2d_slices)
            f_out.create_dataset("sensitivity_map", data=sensitivity_map)
            print(f"Processed: {output_file_name}")

print("\nAll files converted and saved to:", OUTPUT_H5_DIR)
