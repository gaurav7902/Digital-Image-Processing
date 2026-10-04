"""
Lab 7: Image Enhancement in Frequency Domain
Digital Image Processing (NCSH302)

This solution implements:
1. Gaussian Low-Pass Filtering for noise reduction
2. Butterworth High-Pass Filtering for edge enhancement
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================================================================
# PART 1: Gaussian Low-Pass Filter for Noise Reduction
# ==============================================================================

def load_and_resize_image(image_path, target_size=(256, 256)):
    """Load image and resize to target size"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.float64)

def compute_2d_dft(image):
    """Compute 2D Discrete Fourier Transform"""
    # DFT using numpy's FFT
    dft = np.fft.fft2(image)
    return dft

def shift_dft(dft):
    """Shift zero frequency to center"""
    return np.fft.fftshift(dft)

def create_gaussian_lpf(shape, D0):
    """
    Create Gaussian Low-Pass Filter
    H(u,v) = exp[-D^2(u,v) / (2*D0^2)]
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grid
    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    V, U = np.meshgrid(v, u)
    
    # Calculate distance from center
    D = np.sqrt(U**2 + V**2)
    
    # Gaussian LPF formula
    H = np.exp(-D**2 / (2 * D0**2))
    
    return H

def apply_frequency_filter(dft_shifted, filter_mask):
    """Apply filter in frequency domain"""
    return dft_shifted * filter_mask

def compute_inverse_dft(dft_filtered):
    """Compute inverse DFT to get spatial domain image"""
    # Shift back before inverse DFT
    dft_unshifted = np.fft.ifftshift(dft_filtered)
    # Inverse FFT
    image_reconstructed = np.fft.ifft2(dft_unshifted)
    # Take real part (imaginary part is negligible)
    return np.real(image_reconstructed)

def calculate_mse(original, reconstructed):
    """Calculate Mean Squared Error"""
    return np.mean((original - reconstructed) ** 2)

def calculate_psnr(original, reconstructed, max_pixel=255):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    # Normalize images to [0, 1] range for PSNR calculation
    psnr = 10 * np.log10((max_pixel**2) / mse)
    return psnr

def part1_gaussian_lpf():
    """Part 1: Gaussian Low-Pass Filter for noise reduction"""
    print("=" * 60)
    print("PART 1: Gaussian Low-Pass Filter for Noise Reduction")
    print("=" * 60)
    
    # Load images
    noisy_img = load_and_resize_image('gaussian_noise.png')
    clean_img = load_and_resize_image('reference.png')
    
    print(f"Noisy image shape: {noisy_img.shape}")
    print(f"Clean reference shape: {clean_img.shape}")
    
    # Normalize images to [0, 1] range for processing
    noisy_normalized = noisy_img / 255.0
    clean_normalized = clean_img / 255.0
    
    # Step 1: Compute 2D DFT
    dft = compute_2d_dft(noisy_normalized)
    print("Computed 2D DFT of noisy image")
    
    # Step 2: Shift zero frequency to center
    dft_shifted = shift_dft(dft)
    print("Shifted zero frequency to center")
    
    # Step 3: Create Gaussian LPF with D0 = 30
    D0 = 30
    glpf = create_gaussian_lpf(noisy_normalized.shape, D0)
    print(f"Created Gaussian LPF with cutoff D0 = {D0}")
    
    # Step 4: Apply filter
    filtered_dft = apply_frequency_filter(dft_shifted, glpf)
    print("Applied filter to frequency domain")
    
    # Step 5: Inverse DFT to get spatial domain
    denoised_image = compute_inverse_dft(filtered_dft)
    print("Computed inverse DFT to get denoised image")
    
    # Calculate MSE and PSNR
    mse_noisy = calculate_mse(clean_normalized, noisy_normalized)
    mse_denoised = calculate_mse(clean_normalized, denoised_image)
    
    psnr_noisy = 10 * np.log10(1 / mse_noisy) if mse_noisy > 0 else float('inf')
    psnr_denoised = 10 * np.log10(1 / mse_denoised) if mse_denoised > 0 else float('inf')
    
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"{'Image':<20} {'MSE':<15} {'PSNR (dB)':<15}")
    print("-" * 40)
    print(f"{'Noisy image':<20} {mse_noisy:<15.6f} {psnr_noisy:<15.3f}")
    print(f"{'Noise-reduced image':<20} {mse_denoised:<15.6f} {psnr_denoised:<15.3f}")
    print("-" * 40)
    
    # Display images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(noisy_normalized, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    # Show centered Fourier spectrum (log magnitude)
    spectrum = np.log(np.abs(dft_shifted) + 1)
    plt.imshow(spectrum, cmap='gray')
    plt.title('Centered Fourier Spectrum')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(glpf, cmap='gray')
    plt.title(f'Gaussian LPF (D0={D0})')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(denoised_image, cmap='gray')
    plt.title('Noise-Reduced Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('lab7_part1_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved results to lab7_part1_results.png")
    
    return denoised_image, psnr_noisy, psnr_denoised


# ==============================================================================
# PART 2: Butterworth High-Pass Filter for Edge Enhancement
# ==============================================================================

def create_butterworth_hpf(shape, D0, n=2):
    """
    Create Butterworth High-Pass Filter
    H(u,v) = 1 / [1 + (D0/D(u,v))^(2n)]
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grid
    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    V, U = np.meshgrid(v, u)
    
    # Calculate distance from center
    D = np.sqrt(U**2 + V**2)
    
    # Avoid division by zero
    D[D == 0] = 0.0001
    
    # Butterworth HPF formula
    H = 1 / (1 + (D0 / D) ** (2 * n))
    
    return H

def sobel_filter(image):
    """
    Apply Sobel filters manually
    Returns gradient magnitude
    """
    # Sobel kernels
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    
    Gy = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]], dtype=np.float64)
    
    rows, cols = image.shape
    result_x = np.zeros_like(image, dtype=np.float64)
    result_y = np.zeros_like(image, dtype=np.float64)
    
    # Apply convolution manually (without padding)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i-1:i+2, j-1:j+2]
            result_x[i, j] = np.sum(window * Gx)
            result_y[i, j] = np.sum(window * Gy)
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(result_x**2 + result_y**2)
    
    return gradient_magnitude

def calculate_tenengrad(image):
    """Calculate Tenengrad (sharpness measure)"""
    gradient = sobel_filter(image)
    return np.mean(gradient ** 2)

def part2_butterworth_hpf():
    """Part 2: Butterworth High-Pass Filter for edge enhancement"""
    print("\n" + "=" * 60)
    print("PART 2: Butterworth High-Pass Filter for Edge Enhancement")
    print("=" * 60)
    
    # Load and resize image
    img = load_and_resize_image('building.png')
    print(f"Input image shape: {img.shape}")
    
    # Normalize to [0, 1]
    img_normalized = img / 255.0
    
    # Step 1: Compute 2D DFT
    dft = compute_2d_dft(img_normalized)
    print("Computed 2D DFT of image")
    
    # Step 2: Shift zero frequency to center
    dft_shifted = shift_dft(dft)
    print("Shifted zero frequency to center")
    
    # Step 3: Create Butterworth HPF with D0 = 30, n = 2
    D0 = 30
    n = 2
    bhpf = create_butterworth_hpf(img_normalized.shape, D0, n)
    print(f"Created Butterworth HPF with cutoff D0 = {D0}, order n = {n}")
    
    # Step 4: Apply filter
    filtered_dft = apply_frequency_filter(dft_shifted, bhpf)
    print("Applied filter to frequency domain")
    
    # Step 5: Inverse DFT to get spatial domain
    hp_filtered = compute_inverse_dft(filtered_dft)
    print("Computed inverse DFT to get high-pass filtered image")
    
    # Ensure values are in valid range
    hp_filtered = np.clip(hp_filtered, 0, 1)
    
    # Step 6: Apply Sobel filters to both original and filtered images
    print("Applying Sobel filters...")
    
    tenengrad_original = calculate_tenengrad(img_normalized)
    tenengrad_filtered = calculate_tenengrad(hp_filtered)
    
    print("\n" + "-" * 40)
    print("Sharpness Analysis (Tenengrad):")
    print("-" * 40)
    print(f"{'Image':<25} {'Tenengrad (T)':<20}")
    print("-" * 40)
    print(f"{'Original Image':<25} {tenengrad_original:<20.6f}")
    print(f"{'High-Pass Filtered Image':<25} {tenengrad_filtered:<20.6f}")
    print("-" * 40)
    
    if tenengrad_filtered > tenengrad_original:
        print("Edge enhancement successful! Tenengrad increased.")
    else:
        print("Note: Tenengrad decreased, but visual enhancement may still be present.")
    
    # Display images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_normalized, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    spectrum = np.log(np.abs(dft_shifted) + 1)
    plt.imshow(spectrum, cmap='gray')
    plt.title('Centered Fourier Spectrum')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(bhpf, cmap='gray')
    plt.title(f'Butterworth HPF (D0={D0}, n={n})')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(hp_filtered, cmap='gray')
    plt.title('High-Pass Filtered Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('lab7_part2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved results to lab7_part2_results.png")
    
    return hp_filtered, tenengrad_original, tenengrad_filtered


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    # Change to Lab7 directory
    import os
    os.chdir('/home/gaurav/Desktop/Class/Clg/Digital Image Processing/Lab/Lab7')
    
    # Run Part 1: Gaussian LPF
    denoised_img, psnr_noisy, psnr_denoised = part1_gaussian_lpf()
    
    # Run Part 2: Butterworth HPF
    hp_img, ten_orig, ten_filt = part2_butterworth_hpf()
    
    print("\n" + "=" * 60)
    print("LAB 7 COMPLETED SUCCESSFULLY")
    print("=" * 60)