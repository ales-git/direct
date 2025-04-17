import h5py
import numpy as np
import os
import xml.etree.ElementTree as ET

# --- SETTINGS ---
INPUT_H5_DIR = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5/"
OUTPUT_H5_DIR = "/home/giovanni/Desktop/Projects/AI/Projects/direct/projects/cvpr2022_recurrentvarnet/bracco_data_test/h5_adapted_2D/"
os.makedirs(OUTPUT_H5_DIR, exist_ok=True)

# --- GENERATE ISMRMRD HEADER ---
def generate_ismrmrd_header(kspace_shape):
    """
    kspace_shape = (coil, Nz, Ny, Nx)
    """
    root = ET.Element("ismrmrdHeader", xmlns="http://www.ismrm.org/ISMRMRD")
    encoding = ET.SubElement(root, "encoding")

    encoded_space = ET.SubElement(encoding, "encodedSpace")
    matrix_size = ET.SubElement(encoded_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = str(kspace_shape[3])  # x = Nx
    ET.SubElement(matrix_size, "y").text = str(kspace_shape[2])  # y = Ny
    ET.SubElement(matrix_size, "z").text = str(kspace_shape[1])  # z = Nz

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

# --- CONVERT EACH FILE ---
for file_name in os.listdir(INPUT_H5_DIR):
    if not file_name.endswith(".h5"):
        continue

    input_path = os.path.join(INPUT_H5_DIR, file_name)
    output_path = os.path.join(OUTPUT_H5_DIR, file_name)

    with h5py.File(input_path, "r") as f_in:
        kspace_3d = f_in["kspace"][:]  # shape: (Nz, Ny, Nx)
        print(f"Original k-space shape: {kspace_3d.shape}")

        # Apply IFFT along readout (x) to simulate hybrid x-ky (as in 2D slice acquisition)
        kspace_hybrid = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(kspace_3d, axes=(-1,)), axis=-1), axes=(-1,))


        # Add coil dimension: (1, Nz, Ny, Nx)
        kspace_hybrid = np.expand_dims(kspace_hybrid, axis=0).astype(np.complex64)
        print(f"Hybrid 2D k-space volume shape: {kspace_hybrid.shape}")

        # Create dummy sensitivity map (1s)
        sensitivity_map = np.ones_like(kspace_hybrid, dtype=np.complex64)

        # Create ISMRMRD header
        ismrmrd_header = generate_ismrmrd_header(kspace_hybrid.shape)

        # Write HDF5 output
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("kspace", data=kspace_hybrid)
            f_out.create_dataset("sensitivity_map", data=sensitivity_map)
            f_out.create_dataset("ismrmrd_header", data=np.bytes_(ismrmrd_header))
            print(f"Saved: {output_path}")

print("\nAll files processed and saved to:", OUTPUT_H5_DIR)
