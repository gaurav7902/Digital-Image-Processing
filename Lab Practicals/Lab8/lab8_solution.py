"""
Lab 8: Image Restoration
Digital Image Processing (NCSH302)

This solution implements:
1. Restoration from multiplicative noise using logarithmic transformation
2. Median filtering for salt-and-pepper noise and Butterworth band-reject filtering
3. Contraharmonic mean filtering for pepper and salt noise
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_image(image_path):
    """Load grayscale image"""
    img = Image.open(image_path).convert('L')
    return np.array(img, dtype=np.float64)

def calculate_mse(original, degraded):
    """Calculate Mean Squared Error"""
    return np.mean((original - degraded) ** 2)

def calculate_psnr(original, restored, max_pixel=255):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(original, restored)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


# ==============================================================================
# PART 1: Multiplicative Noise Restoration
# ==============================================================================

def part1_multiplicative_noise():
    """Part 1: Restore image degraded by multiplicative noise"""
    print("=" * 70)
    print("PART 1: Multiplicative Noise Restoration")
    print("=" * 70)
    
    # Load original image
    f_original = load_image('Blurry_moon.tif')
    print(f"Original image shape: {f_original.shape}")
    print(f"Original image range: [{f_original.min():.0f}, {f_original.max():.0f}]")
    
    # Generate uniform multiplicative noise
    np.random.seed(0)
    rows, cols = f_original.shape
    noise = np.random.uniform(0.8, 1.2, size=(rows, cols))
    
    # Create degraded image
    g_degraded = f_original * noise
    g_degraded = np.clip(g_degraded, 0, 255).astype(np.uint8)
    g_degraded = g_degraded.astype(np.float64)
    
    print(f"Degraded image range: [{g_degraded.min():.0f}, {g_degraded.max():.0f}]")
    
    # Apply logarithmic transformation
    z = np.log(1 + g_degraded)
    
    # Apply 3x3 arithmetic mean filter manually
    z_filtered = np.zeros_like(z)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = z[i-1:i+2, j-1:j+2]
            z_filtered[i, j] = np.mean(window)
    
    # Handle boundaries (copy values)
    z_filtered[0, :] = z_filtered[1, :]
    z_filtered[-1, :] = z_filtered[-2, :]
    z_filtered[:, 0] = z_filtered[:, 1]
    z_filtered[:, -1] = z_filtered[:, -2]
    
    # Inverse transformation
    f_restored = np.exp(z_filtered) - 1
    f_restored = np.clip(f_restored, 0, 255)
    
    # Calculate MSE values
    mse_degraded = calculate_mse(f_original, g_degraded)
    mse_restored = calculate_mse(f_original, f_restored)
    
    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)
    print(f"MSE (original - degraded) = {mse_degraded:.2f}")
    print(f"MSE (original - restored) = {mse_restored:.2f}")
    print(f"Improvement: {((mse_degraded - mse_restored) / mse_degraded * 100):.2f}%")
    print("-" * 70)
    
    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(f_original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(g_degraded, cmap='gray')
    axes[1].set_title('Degraded (Multiplicative Noise)')
    axes[1].axis('off')
    
    axes[2].imshow(f_restored, cmap='gray')
    axes[2].set_title('Restored Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('lab8_part1_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved results to lab8_part1_results.png")
    
    return f_original, g_degraded, f_restored, mse_degraded, mse_restored


# ==============================================================================
# PART 2: Salt-and-Pepper and Periodic Noise Restoration
# ==============================================================================

def add_salt_pepper_noise(image, pa=0.05, pb=0.05, seed=0):
    """Add salt-and-pepper noise to image"""
    np.random.seed(seed)
    noisy = image.copy()
    rows, cols = image.shape
    
    # Generate random positions for pepper (0) and salt (255)
    noise_mask = np.random.random((rows, cols))
    
    # Add pepper noise
    pepper_mask = noise_mask < pa
    noisy[pepper_mask] = 0
    
    # Add salt noise
    salt_mask = (noise_mask >= pa) & (noise_mask < (pa + pb))
    noisy[salt_mask] = 255
    
    return noisy

def median_filter_3x3(image):
    """Apply 3x3 median filter manually"""
    rows, cols = image.shape
    filtered = np.zeros_like(image)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i-1:i+2, j-1:j+2].flatten()
            filtered[i, j] = np.median(window)
    
    # Handle boundaries
    filtered[0, :] = filtered[1, :]
    filtered[-1, :] = filtered[-2, :]
    filtered[:, 0] = filtered[:, 1]
    filtered[:, -1] = filtered[:, -2]
    
    return filtered

def create_butterworth_band_reject(shape, D0, W, n=4):
    """Create Butterworth Band-Reject Filter"""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grid
    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    V, U = np.meshgrid(v, u)
    
    # Calculate distance from center
    D = np.sqrt(U**2 + V**2)
    
    # Butterworth Band-Reject filter
    H = 1 / (1 + ((D * W) / (D**2 - D0**2)) ** (2 * n))
    H = np.nan_to_num(H, nan=0.0)  # Handle division by zero
    
    return H

def apply_butterworth_band_reject(image, D0, W, n=4):
    """Apply Butterworth band-reject filter"""
    # FFT
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    
    # Create filter
    H = create_butterworth_band_reject(image.shape, D0, W, n)
    
    # Apply filter
    filtered_dft = dft_shifted * H
    
    # Inverse FFT
    dft_unshifted = np.fft.ifftshift(filtered_dft)
    filtered = np.fft.ifft2(dft_unshifted)
    filtered = np.real(filtered)
    
    return filtered

def add_sinusoidal_noise(image, A, u0, v0, seed=0):
    """Add sinusoidal periodic noise"""
    np.random.seed(seed)
    rows, cols = image.shape
    x = np.arange(rows)
    y = np.arange(cols)
    X, Y = np.meshgrid(y, x)
    
    # Sinusoidal noise
    noise = A * np.sin(2 * np.pi * (u0 * X / cols + v0 * Y / rows))
    noisy = image + noise
    
    return np.clip(noisy, 0, 255)

def part2_salt_pepper_and_periodic():
    """Part 2: Salt-and-pepper and periodic noise restoration"""
    print("\n" + "=" * 70)
    print("PART 2: Salt-and-Pepper and Periodic Noise Restoration")
    print("=" * 70)
    
    # Load original image
    f_original = load_image('Blurry_moon.tif')
    
    # Experiment 1: Salt-and-pepper noise with median filter
    print("\nExperiment 1: Salt-and-Pepper Noise + Median Filter")
    print("-" * 70)
    
    g_sp = add_salt_pepper_noise(f_original, pa=0.05, pb=0.05, seed=0)
    f_sp_restored = median_filter_3x3(g_sp)
    
    psnr_sp_noisy = calculate_psnr(f_original, g_sp)
    psnr_sp_restored = calculate_psnr(f_original, f_sp_restored)
    
    print(f"PSNR - Salt-pepper noisy: {psnr_sp_noisy:.2f} dB")
    print(f"PSNR - Median restored: {psnr_sp_restored:.2f} dB")
    print(f"Improvement: {psnr_sp_restored - psnr_sp_noisy:.2f} dB")
    
    # Experiment 2: Sinusoidal noise with Butterworth band-reject filter
    print("\nExperiment 2: Sinusoidal Noise + Butterworth Band-Reject Filter")
    print("-" * 70)
    
    A = 40
    u0 = 0
    v0 = 40
    g_sinusoidal = add_sinusoidal_noise(f_original, A, u0, v0, seed=0)
    f_sin_restored = apply_butterworth_band_reject(g_sinusoidal, D0=40, W=6, n=4)
    f_sin_restored = np.clip(f_sin_restored, 0, 255)
    
    psnr_sin_noisy = calculate_psnr(f_original, g_sinusoidal)
    psnr_sin_restored = calculate_psnr(f_original, f_sin_restored)
    
    print(f"PSNR - Sinusoidal noisy: {psnr_sin_noisy:.2f} dB")
    print(f"PSNR - Butterworth restored: {psnr_sin_restored:.2f} dB")
    print(f"Improvement: {psnr_sin_restored - psnr_sin_noisy:.2f} dB")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(g_sp, cmap='gray')
    axes[0, 0].set_title(f'Salt-Pepper Noisy\nPSNR: {psnr_sp_noisy:.2f} dB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(f_sp_restored, cmap='gray')
    axes[0, 1].set_title(f'Median Restored\nPSNR: {psnr_sp_restored:.2f} dB')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(g_sinusoidal, cmap='gray')
    axes[1, 0].set_title(f'Sinusoidal Noisy\nPSNR: {psnr_sin_noisy:.2f} dB')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(f_sin_restored, cmap='gray')
    axes[1, 1].set_title(f'Butterworth Restored\nPSNR: {psnr_sin_restored:.2f} dB')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('lab8_part2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved results to lab8_part2_results.png")
    
    return psnr_sp_noisy, psnr_sp_restored, psnr_sin_noisy, psnr_sin_restored


# ==============================================================================
# PART 3: Contraharmonic Mean Filtering
# ==============================================================================

def contraharmonic_mean_filter(image, kernel_size, Q):
    """
    Apply contraharmonic mean filter
    f̂(x,y) = Σ[g(s,t)^(Q+1)] / Σ[g(s,t)^Q]
    """
    rows, cols = image.shape
    filtered = np.zeros_like(image)
    pad = kernel_size // 2
    
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            window = image[i-pad:i+pad+1, j-pad:j+pad+1]
            
            # Handle zero values to avoid issues with negative powers
            if Q < 0:
                # For negative Q, avoid division by zero
                window = window.copy()
                window[window == 0] = 1e-10
            
            numerator = np.sum(window ** (Q + 1))
            denominator = np.sum(window ** Q)
            
            if denominator != 0:
                filtered[i, j] = numerator / denominator
            else:
                filtered[i, j] = window[pad, pad]
    
    # Handle boundaries
    filtered = np.clip(filtered, 0, 255)
    
    return filtered

def add_pepper_noise(image, percentage, seed=0):
    """Add pepper noise (set pixels to 0)"""
    np.random.seed(seed)
    noisy = image.copy()
    rows, cols = image.shape
    
    n_pepper = int(rows * cols * percentage)
    pepper_indices = np.random.choice(rows * cols, n_pepper, replace=False)
    
    noisy.flat[pepper_indices] = 0
    
    return noisy

def add_salt_noise(image, percentage, seed=0):
    """Add salt noise (set pixels to 255)"""
    np.random.seed(seed)
    noisy = image.copy()
    rows, cols = image.shape
    
    n_salt = int(rows * cols * percentage)
    salt_indices = np.random.choice(rows * cols, n_salt, replace=False)
    
    noisy.flat[salt_indices] = 255
    
    return noisy

def part3_contraharmonic_mean():
    """Part 3: Contraharmonic mean filtering for pepper and salt noise"""
    print("\n" + "=" * 70)
    print("PART 3: Contraharmonic Mean Filtering")
    print("=" * 70)
    
    # Load original image
    f_original = load_image('Circuit_Board.tif')
    print(f"Original image shape: {f_original.shape}")
    
    # Add pepper noise (10%)
    print("\nPepper Noise Restoration (Q = +1.5):")
    print("-" * 70)
    
    g_pepper = add_pepper_noise(f_original, 0.10, seed=0)
    f_pepper_restored = contraharmonic_mean_filter(g_pepper, kernel_size=5, Q=1.5)
    f_pepper_restored = np.clip(f_pepper_restored, 0, 255)
    
    psnr_pepper_noisy = calculate_psnr(f_original, g_pepper)
    psnr_pepper_restored = calculate_psnr(f_original, f_pepper_restored)
    
    print(f"PSNR - Pepper noisy: {psnr_pepper_noisy:.2f} dB")
    print(f"PSNR - Pepper restored: {psnr_pepper_restored:.2f} dB")
    print(f"Improvement: {psnr_pepper_restored - psnr_pepper_noisy:.2f} dB")
    
    # Add salt noise (10%)
    print("\nSalt Noise Restoration (Q = -1.5):")
    print("-" * 70)
    
    g_salt = add_salt_noise(f_original, 0.10, seed=0)
    f_salt_restored = contraharmonic_mean_filter(g_salt, kernel_size=5, Q=-1.5)
    f_salt_restored = np.clip(f_salt_restored, 0, 255)
    
    psnr_salt_noisy = calculate_psnr(f_original, g_salt)
    psnr_salt_restored = calculate_psnr(f_original, f_salt_restored)
    
    print(f"PSNR - Salt noisy: {psnr_salt_noisy:.2f} dB")
    print(f"PSNR - Salt restored: {psnr_salt_restored:.2f} dB")
    print(f"Improvement: {psnr_salt_restored - psnr_salt_noisy:.2f} dB")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(g_pepper, cmap='gray')
    axes[0, 0].set_title(f'Pepper Noisy (10%)\nPSNR: {psnr_pepper_noisy:.2f} dB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(f_pepper_restored, cmap='gray')
    axes[0, 1].set_title(f'Pepper Restored (Q=+1.5)\nPSNR: {psnr_pepper_restored:.2f} dB')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(g_salt, cmap='gray')
    axes[1, 0].set_title(f'Salt Noisy (10%)\nPSNR: {psnr_salt_noisy:.2f} dB')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(f_salt_restored, cmap='gray')
    axes[1, 1].set_title(f'Salt Restored (Q=-1.5)\nPSNR: {psnr_salt_restored:.2f} dB')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('lab8_part3_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved results to lab8_part3_results.png")
    
    return psnr_pepper_noisy, psnr_pepper_restored, psnr_salt_noisy, psnr_salt_restored


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    import os
    os.chdir('/home/gaurav/Desktop/Class/Clg/Digital Image Processing/Lab/Lab8')
    
    # Part 1
    f_orig_1, g_deg_1, f_rest_1, mse_deg_1, mse_rest_1 = part1_multiplicative_noise()
    
    # Part 2
    psnr_sp_n, psnr_sp_r, psnr_sin_n, psnr_sin_r = part2_salt_pepper_and_periodic()
    
    # Part 3
    psnr_pp_n, psnr_pp_r, psnr_ps_n, psnr_ps_r = part3_contraharmonic_mean()
    
    print("\n" + "=" * 70)
    print("LAB 8 COMPLETED SUCCESSFULLY")
    print("=" * 70)
