"""
Lab 4: Image Enhancement-I (Pixel Geometry-II & Spatial Domain)
Digital Image Processing (NCSH302)

This solution implements:
1. Region adjacency and boundary extraction
2. Foreground/background separation
3. Histogram equalization
4. Spatial filtering (smoothing and sharpening)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================================================================
# PART 1: Region Operations
# ==============================================================================

def find_connected_components(binary_image, connectivity=8):
    """Find connected components using flood fill"""
    labeled = np.zeros_like(binary_image, dtype=int)
    label = 0
    rows, cols = binary_image.shape
    
    def flood_fill(i, j, label):
        """Flood fill from position (i, j)"""
        stack = [(i, j)]
        while stack:
            x, y = stack.pop()
            if labeled[x, y] != 0:
                continue
            labeled[x, y] = label
            
            # Add neighbors based on connectivity
            neighbors = []
            if connectivity == 4:
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            else:  # 8-connectivity
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                           (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
            
            for nx, ny in neighbors:
                if (0 <= nx < rows and 0 <= ny < cols and 
                    binary_image[nx, ny] == 1 and labeled[nx, ny] == 0):
                    stack.append((nx, ny))
    
    # Find all components
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1 and labeled[i, j] == 0:
                label += 1
                flood_fill(i, j, label)
    
    return labeled, label

def region_adjacency(binary_image, Ri, Rj):
    """
    Check if regions Ri and Rj are adjacent
    Returns 1 if adjacent, 0 if disjoint
    """
    # Find pixels in each region
    mask_i = (binary_image == Ri)
    mask_j = (binary_image == Rj)
    
    rows, cols = binary_image.shape
    
    # Check if any pixel in Ri has a neighbor in Rj
    for i in range(rows):
        for j in range(cols):
            if mask_i[i, j]:
                # Check 8-neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and mask_j[ni, nj]):
                            return 1
    return 0

def boundary_extract(binary_image, connectivity=8):
    """
    Extract boundary of each region
    Boundary = Region - Eroded(Region)
    """
    labeled, num_regions = find_connected_components(binary_image, connectivity)
    boundary_image = np.zeros_like(binary_image)
    boundary_counts = {}
    
    rows, cols = binary_image.shape
    
    for region_label in range(1, num_regions + 1):
        region_mask = (labeled == region_label)
        
        # Find boundary pixels
        for i in range(rows):
            for j in range(cols):
                if region_mask[i, j]:
                    # Check if any neighbor is background
                    is_boundary = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if not (0 <= ni < rows and 0 <= nj < cols) or binary_image[ni, nj] == 0:
                                is_boundary = True
                                break
                        if is_boundary:
                            break
                    
                    if is_boundary:
                        boundary_image[i, j] = 1
        
        boundary_counts[region_label] = np.sum((labeled == region_label) & (boundary_image == 1))
    
    return boundary_image, boundary_counts

def foreground_background(binary_image):
    """
    Separate foreground and background
    Returns foreground image, background image, and pixel counts
    """
    foreground = binary_image.copy()
    background = 1 - binary_image
    
    fg_count = np.sum(foreground)
    bg_count = np.sum(background)
    
    return foreground, background, fg_count, bg_count

def part1_region_operations():
    """Part 1: Region operations"""
    print("=" * 70)
    print("PART 1: Region Operations")
    print("=" * 70)
    
    # Create sample binary image
    binary_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    
    print("Binary Image:")
    print(binary_image)
    
    # Find connected regions
    labeled, num_regions = find_connected_components(binary_image, connectivity=8)
    print(f"\nNumber of connected regions = {num_regions}")
    
    # Extract boundaries
    boundary, boundary_counts = boundary_extract(binary_image, connectivity=8)
    print("\nBoundary pixel counts per region:")
    for region, count in boundary_counts.items():
        print(f"  Region {region}: {count} boundary pixels")
    
    # Foreground/background
    fg, bg, fg_count, bg_count = foreground_background(binary_image)
    print(f"\nForeground pixels: {fg_count}")
    print(f"Background pixels: {bg_count}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(binary_image, cmap='gray')
    axes[0, 0].set_title('Original Binary Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(labeled, cmap='tab10')
    axes[0, 1].set_title(f'Connected Components ({num_regions})')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(boundary, cmap='gray')
    axes[1, 0].set_title('Boundary Extraction')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(bg, cmap='gray')
    axes[1, 1].set_title('Background Image')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('lab4_part1_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return binary_image, labeled, boundary

# ==============================================================================
# PART 2: Histogram Equalization
# ==============================================================================

def compute_histogram(image):
    """Compute histogram of grayscale image"""
    hist = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        hist[int(pixel)] += 1
    return hist

def histogram_equalization(image):
    """
    Perform histogram equalization
    sk = (L-1) * sum(pr(rj)) for j=0 to k
    """
    rows, cols = image.shape
    N = rows * cols
    L = 256
    
    # Compute histogram
    hist = compute_histogram(image)
    
    # Compute PDF
    pdf = hist / N
    
    # Compute CDF
    cdf = np.zeros(256)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    
    # Transformation function
    transform = np.round((L - 1) * cdf).astype(np.uint8)
    
    # Apply transformation
    equalized = transform[image]
    
    return equalized, hist, transform

# ==============================================================================
# PART 3: Spatial Filtering
# ==============================================================================

def mean_filter(image, kernel_size):
    """Apply mean (averaging) filter"""
    rows, cols = image.shape
    pad = kernel_size // 2
    filtered = np.zeros_like(image, dtype=np.float64)
    
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            window = image[i-pad:i+pad+1, j-pad:j+pad+1]
            filtered[i, j] = np.mean(window)
    
    # Handle boundaries
    filtered[:pad, :] = image[:pad, :]
    filtered[-pad:, :] = image[-pad:, :]
    filtered[:, :pad] = image[:, :pad]
    filtered[:, -pad:] = image[:, -pad:]
    
    return filtered.astype(np.uint8)

def median_filter(image, kernel_size):
    """Apply median filter"""
    rows, cols = image.shape
    pad = kernel_size // 2
    filtered = np.zeros_like(image)
    
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            window = image[i-pad:i+pad+1, j-pad:j+pad+1]
            filtered[i, j] = np.median(window)
    
    # Handle boundaries
    filtered[:pad, :] = image[:pad, :]
    filtered[-pad:, :] = image[-pad:, :]
    filtered[:, :pad] = image[:, :pad]
    filtered[:, -pad:] = image[:, -pad:]
    
    return filtered

def laplacian_sharpening(image):
    """Apply Laplacian sharpening"""
    # Laplacian kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float64)
    
    rows, cols = image.shape
    laplacian = np.zeros_like(image, dtype=np.float64)
    
    # Apply Laplacian
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i-1:i+2, j-1:j+2]
            laplacian[i, j] = np.sum(window * kernel)
    
    # Sharpen: g(x,y) = f(x,y) - ∇²f(x,y)
    sharpened = image.astype(np.float64) - laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import os
    os.chdir('/home/gaurav/Desktop/Class/Clg/Digital Image Processing/Lab/Lab4')
    
    print("Lab 4: Image Enhancement-I")
    print("=" * 70)
    
    # Part 1: Region operations
    binary_img, labeled_img, boundary_img = part1_region_operations()
    
    # Part 2 & 3: If images are available
    try:
        # Load image
        input_img = np.array(Image.open('input.png').convert('L'))
        
        print("\n" + "=" * 70)
        print("PART 2 & 3: Histogram Equalization and Spatial Filtering")
        print("=" * 70)
        
        # Histogram equalization
        eq_img, hist_orig, transform = histogram_equalization(input_img)
        
        # Spatial filtering
        mean_3 = mean_filter(input_img, 3)
        median_3 = median_filter(input_img, 3)
        sharpened = laplacian_sharpening(input_img)
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(eq_img, cmap='gray')
        axes[0, 1].set_title('Histogram Equalized')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mean_3, cmap='gray')
        axes[0, 2].set_title('Mean Filter (3x3)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(median_3, cmap='gray')
        axes[1, 0].set_title('Median Filter (3x3)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(sharpened, cmap='gray')
        axes[1, 1].set_title('Laplacian Sharpened')
        axes[1, 1].axis('off')
        
        # Histogram comparison
        hist_eq = compute_histogram(eq_img)
        axes[1, 2].plot(hist_orig, 'b', label='Original')
        axes[1, 2].plot(hist_eq, 'r', label='Equalized')
        axes[1, 2].set_title('Histograms')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('lab4_part2_3_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Histogram equalization and spatial filtering completed")
        
    except FileNotFoundError:
        print("\nNote: input.png not found. Part 2 & 3 skipped.")
    
    print("\n" + "=" * 70)
    print("LAB 4 COMPLETED SUCCESSFULLY")
    print("=" * 70)
