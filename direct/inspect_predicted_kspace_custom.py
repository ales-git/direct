
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the saved tensor
output_kspace = torch.load("/home/giovanni/Desktop/Projects/AI/Projects/direct/direct/predicted_kspace.pt")

# Assuming output_kspace has shape [1, 16, 256, 256, 2]
# Extract the real and imaginary parts from k-space
real_part = output_kspace[..., 0]  # shape: [1, 16, 256, 256]
imag_part = output_kspace[..., 1]  # shape: [1, 16, 256, 256]

# Combine real and imaginary parts into a complex tensor
complex_kspace = torch.complex(real_part, imag_part)  # shape: [1, 16, 256, 256]

# Apply FFT shift (same as np.fft.fftshift)
kspace_shifted = torch.fft.fftshift(complex_kspace, dim=(-2, -1))  # shift along height and width

# Apply rolling (same as np.roll)
pixshift = 12
pixshift2 = -26
kspace_rolled = torch.roll(kspace_shifted, shifts=(pixshift, pixshift2), dims=(2, 3))

# Move the axes as required (like np.moveaxis)
kspace_final = kspace_rolled.permute(0, 1, 3, 2)  # shape: [1, 16, 256, 256]

# Perform Inverse FFT (3D inverse FFT on the final k-space)
image_reconstructed = torch.fft.ifftn(kspace_final, dim=(-2, -1))  # Inverse FFT on last two dims

# Take the magnitude (abs) to get the final image
image_reconstructed = torch.abs(image_reconstructed)

# Now we need to assemble the entire volume from these 16 coils:
# Assuming you're summing across coils (16 coils)
#image_volume = torch.sqrt(torch.sum(image_reconstructed**2, dim=1))  # shape: [1, 256, 256]

# Squeeze batch dimension
image_volume = image_reconstructed.squeeze(0)  # Now shape: [16, 256, 256]

# Save individual slices
output_dir = 'reconstructed_images'
os.makedirs(output_dir, exist_ok=True)

for i in range(image_volume.shape[0]):
    slice_img = image_volume[i, :, :].cpu().numpy()  # Get slice as NumPy array
    slice_img = np.flip(slice_img, axis=0)  # Flip to correct orientation if needed
    slice_filename = os.path.join(output_dir, f'slice_{i}.png')
    plt.imsave(slice_filename, slice_img, cmap='gray')  # Save slice as image

# Optionally, you can save the full 3D volume as a single file (e.g., using np.save)
volume_numpy = image_volume.cpu().numpy()  # Convert to NumPy array for saving
np.save('reconstructed_volume.npy', volume_numpy)  # Save the entire volume as a .npy file