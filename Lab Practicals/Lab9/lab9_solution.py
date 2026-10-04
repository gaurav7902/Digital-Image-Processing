"""
Lab 9: Image Morphology
Digital Image Processing (NCSH302)

This solution implements:
1. Erosion, Dilation, and Boundary Extraction
2. Morphological connected-component extraction and hole identification
3. Morphological region filling
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_binary_image(image_path):
    """Load binary image"""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.uint8)
    # Convert to binary (0 or 1)
    binary = (img_array > 127).astype(np.uint8)
    return binary

def create_5x5_square_structuring_element():
    """Create 5x5 square structuring element with all 1's"""
    return np.ones((5, 5), dtype=np.uint8)

def create_3x3_cross_structuring_element():
    """Create 3x3 cross-shaped structuring element"""
    B = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8)
    return B


# ==============================================================================
# PART 1: Erosion, Dilation, and Boundary Extraction
# ==============================================================================

def erosion(image, structuring_element):
    """
    Erosion operation: output pixel = min of neighborhood
    For binary images: output = 1 only if all pixels in SE are 1
    """
    rows, cols = image.shape
    se_rows, se_cols = structuring_element.shape
    pad_r, pad_c = se_rows // 2, se_cols // 2
    
    eroded = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(pad_r, rows - pad_r):
        for j in range(pad_c, cols - pad_c):
            window = image[i-pad_r:i+pad_r+1, j-pad_c:j+pad_c+1]
            # Erosion: all SE positions must have 1
            if np.all(window[structuring_element == 1] == 1):
                eroded[i, j] = 1
    
    return eroded

def dilation(image, structuring_element):
    """
    Dilation operation: output pixel = max of neighborhood
    For binary images: output = 1 if any pixel in SE is 1
    """
    rows, cols = image.shape
    se_rows, se_cols = structuring_element.shape
    pad_r, pad_c = se_rows // 2, se_cols // 2
    
    dilated = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(pad_r, rows - pad_r):
        for j in range(pad_c, cols - pad_c):
            window = image[i-pad_r:i+pad_r+1, j-pad_c:j+pad_c+1]
            # Dilation: if any SE position has 1
            if np.any(window[structuring_element == 1] == 1):
                dilated[i, j] = 1
    
    return dilated

def boundary_extraction(image, structuring_element):
    """
    Boundary extraction: B(A) = A - (A eroded by SE)
    """
    eroded = erosion(image, structuring_element)
    boundary = image - eroded
    return boundary

def part1_erosion_dilation_boundary():
    """Part 1: Erosion, Dilation, and Boundary Extraction"""
    print("=" * 70)
    print("PART 1: Erosion, Dilation, and Boundary Extraction")
    print("=" * 70)
    
    # Load binary image
    A = load_binary_image('alphabet.png')
    print(f"Original image shape: {A.shape}")
    print(f"Foreground pixels (1): {np.sum(A)}")
    print(f"Background pixels (0): {np.sum(A == 0)}")
    
    # Create 5x5 square structuring element
    B = create_5x5_square_structuring_element()
    
    # Apply erosion and dilation
    print("\nApplying morphological operations...")
    eroded = erosion(A, B)
    dilated = dilation(A, B)
    boundary = boundary_extraction(A, B)
    
    print(f"Eroded image foreground pixels: {np.sum(eroded)}")
    print(f"Dilated image foreground pixels: {np.sum(dilated)}")
    print(f"Boundary pixels: {np.sum(boundary)}")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(A, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(eroded, cmap='gray')
    axes[0, 1].set_title('Eroded Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(dilated, cmap='gray')
    axes[1, 0].set_title('Dilated Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(boundary, cmap='gray')
    axes[1, 1].set_title('Boundary Extraction')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('lab9_part1_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved results to lab9_part1_results.png")
    
    return A, eroded, dilated, boundary


# ==============================================================================
# PART 2 & 3: Morphological Connected-Component Extraction and Hole Filling
# ==============================================================================

def morphological_dilate_for_connectivity(image, structuring_element, mask):
    """
    Conditional dilation: dilate only within mask
    Xk = (Xk-1 ⊕ B) ∩ Ac
    """
    rows, cols = image.shape
    se_rows, se_cols = structuring_element.shape
    pad_r, pad_c = se_rows // 2, se_cols // 2
    
    dilated = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(pad_r, rows - pad_r):
        for j in range(pad_c, cols - pad_c):
            window = image[i-pad_r:i+pad_r+1, j-pad_c:j+pad_c+1]
            # Dilation
            if np.any(window[structuring_element == 1] == 1):
                dilated[i, j] = 1
    
    # Apply mask (intersection with mask)
    dilated = dilated & mask
    return dilated

def find_connected_components_morphological(background, structuring_element):
    """
    Find connected components using morphological operations
    Xk = (Xk-1 ⊕ B) ∩ Ac, iterate until convergence
    """
    rows, cols = background.shape
    processed = np.zeros_like(background, dtype=bool)
    components = []
    component_labels = np.zeros_like(background, dtype=int)
    
    # Find unprocessed background pixels
    for i in range(rows):
        for j in range(cols):
            if background[i, j] == 1 and not processed[i, j]:
                # Initialize with this seed pixel
                X_current = np.zeros_like(background, dtype=np.uint8)
                X_current[i, j] = 1
                
                # Iterate until convergence
                iteration = 0
                max_iterations = 1000
                
                while iteration < max_iterations:
                    X_prev = X_current.copy()
                    
                    # Conditional dilation: (Xk-1 ⊕ B) ∩ Ac
                    X_current = morphological_dilate_for_connectivity(
                        X_current, structuring_element, background
                    )
                    
                    # Check convergence
                    if np.array_equal(X_current, X_prev):
                        break
                    
                    iteration += 1
                
                # Mark as processed and store component
                component_pixels = X_current == 1
                processed[component_pixels] = True
                components.append(X_current)
                component_labels[component_pixels] = len(components)
    
    return components, component_labels

def identify_holes(components, background_original, rows, cols):
    """
    Identify holes: background components that don't touch boundary
    """
    holes = []
    hole_positions = []
    hole_components = []
    
    for idx, component in enumerate(components):
        # Check if component touches boundary
        touches_boundary = (
            np.any(component[0, :] == 1) or  # Top
            np.any(component[-1, :] == 1) or  # Bottom
            np.any(component[:, 0] == 1) or  # Left
            np.any(component[:, -1] == 1)    # Right
        )
        
        if not touches_boundary:
            # This is a hole
            holes.append(component)
            hole_components.append(idx)
            
            # Find a seed pixel inside the hole
            positions = np.where(component == 1)
            if len(positions[0]) > 0:
                seed_row, seed_col = positions[0][0], positions[1][0]
                hole_positions.append((seed_row, seed_col))
    
    return holes, hole_positions, hole_components

def part2_hole_identification():
    """Part 2: Morphological connected-component extraction and hole identification"""
    print("\n" + "=" * 70)
    print("PART 2: Connected-Component Extraction and Hole Identification")
    print("=" * 70)
    
    # Load binary image
    A = load_binary_image('region_filling.png')
    print(f"Original image shape: {A.shape}")
    print(f"Foreground (ring) pixels: {np.sum(A)}")
    
    # Compute background
    Ac = 1 - A
    print(f"Background pixels: {np.sum(Ac)}")
    
    # Create 3x3 cross structuring element
    B = create_3x3_cross_structuring_element()
    
    # Find connected components
    print("\nFinding connected components...")
    components, labels = find_connected_components_morphological(Ac, B)
    
    print(f"Total background components found: {len(components)}")
    
    # Identify holes
    rows, cols = A.shape
    holes, hole_seeds, hole_comp_indices = identify_holes(
        components, Ac, rows, cols
    )
    
    print(f"Total holes identified: {len(holes)}")
    print("\nHole Information:")
    print("-" * 70)
    print(f"{'Hole':<8} {'Pixels':<10} {'Seed Position':<20}")
    print("-" * 70)
    
    for h_idx, (hole, seed) in enumerate(zip(holes, hole_seeds)):
        hole_pixels = np.sum(hole)
        print(f"H{h_idx+1:<7} {hole_pixels:<10} {seed}")
    
    print("-" * 70)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(A, cmap='gray')
    axes[0, 0].set_title('Original Image (Rings)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(Ac, cmap='gray')
    axes[0, 1].set_title('Background Image')
    axes[0, 1].axis('off')
    
    # Show external background component
    external_bg = components[0]  # First component usually touches boundary
    axes[1, 0].imshow(external_bg, cmap='gray')
    axes[1, 0].set_title('External Background Component')
    axes[1, 0].axis('off')
    
    # Show all holes with labels
    holes_visualization = np.zeros_like(A, dtype=int)
    for h_idx, hole in enumerate(holes):
        holes_visualization[hole == 1] = h_idx + 1
    
    im = axes[1, 1].imshow(holes_visualization, cmap='tab20')
    axes[1, 1].set_title(f'Identified Holes (Total: {len(holes)})')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('lab9_part2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved results to lab9_part2_results.png")
    
    return A, Ac, holes, hole_seeds


# ==============================================================================
# PART 3: Morphological Region Filling
# ==============================================================================

def morphological_region_fill(Ac, holes, hole_seeds, structuring_element, max_iterations=1000):
    """
    Fill holes using morphological region filling
    Yk = (Yk-1 ⊕ B) ∩ Ac, iterate until convergence
    """
    filled_regions = []
    fill_iterations = []
    
    for hole_idx, seed in enumerate(hole_seeds):
        # Initialize with seed pixel
        Y_current = np.zeros_like(Ac, dtype=np.uint8)
        Y_current[seed[0], seed[1]] = 1
        
        iteration = 0
        iterations_list = [Y_current.copy()]
        
        while iteration < max_iterations:
            Y_prev = Y_current.copy()
            
            # Conditional dilation: (Yk-1 ⊕ B) ∩ Ac
            Y_current = morphological_dilate_for_connectivity(
                Y_current, structuring_element, Ac
            )
            
            iterations_list.append(Y_current.copy())
            
            # Check convergence
            if np.array_equal(Y_current, Y_prev):
                break
            
            iteration += 1
        
        filled_regions.append(Y_current)
        fill_iterations.append(iteration + 1)
    
    return filled_regions, fill_iterations, iterations_list

def part3_region_filling():
    """Part 3: Morphological region filling"""
    print("\n" + "=" * 70)
    print("PART 3: Morphological Region Filling")
    print("=" * 70)
    
    # Load and prepare
    A = load_binary_image('region_filling.png')
    Ac = 1 - A
    B = create_3x3_cross_structuring_element()
    
    # Get holes and seeds from Part 2
    components, labels = find_connected_components_morphological(Ac, B)
    holes, hole_seeds, hole_comp_indices = identify_holes(
        components, Ac, A.shape[0], A.shape[1]
    )
    
    print(f"Total holes to fill: {len(holes)}")
    
    # Perform region filling
    filled_regions, iterations_list, all_iterations = morphological_region_fill(
        Ac, holes, hole_seeds, B
    )
    
    # Calculate filled image
    R = A.copy().astype(np.uint8)
    total_pixels_added = 0
    
    print("\nRegion Filling Results:")
    print("-" * 70)
    print(f"{'Hole':<8} {'Pixels':<10} {'Iterations':<15}")
    print("-" * 70)
    
    for h_idx, (filled_region, iter_count) in enumerate(zip(filled_regions, iterations_list)):
        hole_pixels = np.sum(filled_region)
        R = np.maximum(R, filled_region)
        total_pixels_added += hole_pixels
        print(f"H{h_idx+1:<7} {hole_pixels:<10} {iter_count:<15}")
    
    print("-" * 70)
    
    original_pixels = np.sum(A)
    final_pixels = np.sum(R)
    
    print(f"\nTotal holes filled: {len(holes)}")
    print(f"|A| = {original_pixels}, |R| = {final_pixels}")
    print(f"|R| - |A| = {total_pixels_added}")
    print(f"Total pixels added to A → R: {total_pixels_added}")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(A, cmap='gray')
    axes[0, 0].set_title(f'Original Image\nPixels: {original_pixels}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(Ac, cmap='gray')
    axes[0, 1].set_title('Background Image')
    axes[0, 1].axis('off')
    
    # Show intermediate iterations for first hole
    if len(all_iterations) > 0 and len(all_iterations[0]) > 3:
        iter_vis = all_iterations[0][min(3, len(all_iterations[0])-1)]
        axes[1, 0].imshow(iter_vis, cmap='gray')
        axes[1, 0].set_title(f'First Hole - Intermediate Iteration')
        axes[1, 0].axis('off')
    
    axes[1, 1].imshow(R, cmap='gray')
    axes[1, 1].set_title(f'Final Filled Image\nPixels: {final_pixels}')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('lab9_part3_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved results to lab9_part3_results.png")
    
    return A, R, filled_regions, iterations_list


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    import os
    os.chdir('/home/gaurav/Desktop/Class/Clg/Digital Image Processing/Lab/Lab9')
    
    print("Starting Lab 9: Image Morphology")
    print("=" * 70)
    
    # Part 1: Erosion, Dilation, Boundary
    A_part1, eroded, dilated, boundary = part1_erosion_dilation_boundary()
    
    # Part 2: Hole Identification
    A_part2, Ac_part2, holes, hole_seeds = part2_hole_identification()
    
    # Part 3: Region Filling
    A_part3, R_filled, filled_regions, iterations = part3_region_filling()
    
    print("\n" + "=" * 70)
    print("LAB 9 COMPLETED SUCCESSFULLY")
    print("=" * 70)
