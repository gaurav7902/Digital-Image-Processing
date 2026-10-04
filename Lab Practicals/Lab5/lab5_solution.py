"""
Lab 5: Image Enhancement-III (Advanced Spatial Operations)
Digital Image Processing (NCSH302)

This solution implements:
1. Defect detection using adaptive thresholding
2. Block-based defect analysis
3. Enhanced defect visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================================================================
# DEFECT DETECTION AND ANALYSIS
# ==============================================================================


def detect_defects(image):
    """
    Detect defects in grayscale image
    T = Imin + (1/3) * (Imax - Imin)
    """
    Imin = image.min()
    Imax = image.max()

    # Calculate threshold
    T = Imin + (1 / 3) * (Imax - Imin)

    print(f"Image intensity range: [{Imin:.0f}, {Imax:.0f}]")
    print(f"Threshold T = {T:.2f}")

    # Create binary defect mask
    # B(x,y) = 1 if Imin <= f(x,y) <= T
    defect_mask = ((image >= Imin) & (image <= T)).astype(np.uint8)

    # Create enhanced image
    # Defect pixels = 255, background pixels = original
    enhanced = image.copy()
    enhanced[defect_mask == 1] = 255

    return defect_mask, enhanced, T


def analyze_blocks(defect_mask):
    """
    Analyze 3x3 blocks for complete defects
    S(i,j) = sum of 3x3 block starting at (i,j)
    D(i,j) = 1 if S(i,j) = 9, else 0
    """
    rows, cols = defect_mask.shape

    # Calculate block sums (no padding, so output is (H-2) x (W-2))
    block_rows = rows - 2
    block_cols = cols - 2

    S = np.zeros((block_rows, block_cols), dtype=int)
    D = np.zeros((block_rows, block_cols), dtype=int)

    for i in range(block_rows):
        for j in range(block_cols):
            # Sum of 3x3 block
            block = defect_mask[i : i + 3, j : j + 3]
            S[i, j] = np.sum(block)

            # Mark as completely defective if all 9 pixels are defects
            if S[i, j] == 9:
                D[i, j] = 1

    # Count overlapping defective blocks
    N_blocks = np.sum(D)

    return S, D, N_blocks


def part1_defect_detection():
    """Part 1: Defect detection and block analysis"""
    print("=" * 70)
    print("Lab 5: Defect Detection and Block Analysis")
    print("=" * 70)

    # Sample 8x8 grayscale image
    image = np.array(
        [
            [30, 35, 40, 45, 50, 100, 110, 120],
            [25, 32, 38, 44, 52, 105, 115, 125],
            [20, 28, 34, 42, 48, 110, 120, 130],
            [22, 29, 36, 43, 55, 115, 125, 135],
            [100, 110, 120, 130, 140, 120, 130, 140],
            [105, 115, 125, 135, 145, 125, 135, 145],
            [110, 120, 130, 140, 150, 130, 30, 40],
            [115, 125, 135, 145, 150, 135, 45, 55],
        ],
        dtype=np.float64,
    )

    print(f"Image shape: {image.shape}")
    print("\nOriginal Image:")
    print(image.astype(int))

    # Detect defects
    print("\n" + "-" * 70)
    defect_mask, enhanced, T = detect_defects(image)

    print(f"\nDefect pixels detected: {np.sum(defect_mask)}")

    print("\nBinary Defect Mask B(x,y):")
    print(defect_mask)

    # Analyze blocks
    print("\n" + "-" * 70)
    print("Block Analysis:")
    print("-" * 70)

    S, D, N_blocks = analyze_blocks(defect_mask)

    print(f"Block sum matrix S shape: {S.shape}")
    print(f"\nCompletely defective blocks (D(i,j) = 1):")
    defective_positions = np.where(D == 1)
    if len(defective_positions[0]) > 0:
        for idx in range(len(defective_positions[0])):
            i, j = defective_positions[0][idx], defective_positions[1][idx]
            print(f"  Block at ({i}, {j}): S = {S[i, j]}")
    else:
        print("  No completely defective blocks found")

    print(f"\nTotal overlapping defective blocks: N_blocks = {N_blocks}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(defect_mask, cmap="gray")
    axes[0, 1].set_title(f"Binary Defect Mask\n(T={T:.2f})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(enhanced, cmap="gray")
    axes[0, 2].set_title("Enhanced Image\n(Defects=255)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(S, cmap="hot")
    axes[1, 0].set_title("Block Sum S(i,j)")
    axes[1, 0].axis("off")
    cbar1 = plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0])
    cbar1.set_label("Sum")

    axes[1, 1].imshow(D, cmap="gray")
    axes[1, 1].set_title(f"Defective Blocks D(i,j)\n(N={N_blocks})")
    axes[1, 1].axis("off")

    # Show defect statistics
    axes[1, 2].axis("off")
    stats_text = f"""
    DEFECT ANALYSIS SUMMARY
    {'='*30}
    
    Image Size: {image.shape[0]} x {image.shape[1]}
    
    Threshold: T = {T:.2f}
    
    Total Defect Pixels: {np.sum(defect_mask)}
    Defect Percentage: {np.sum(defect_mask)/(image.size)*100:.1f}%
    
    Block Analysis (3x3):
    - Total blocks: {S.size}
    - Defective blocks: {N_blocks}
    - Defect block %: {N_blocks/S.size*100 if S.size > 0 else 0:.1f}%
    
    Intensity Range:
    - Min: {image.min():.0f}
    - Max: {image.max():.0f}
    - Mean: {image.mean():.1f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family="monospace", va="center")

    plt.tight_layout()
    plt.savefig("lab5_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 70)
    print("Results saved to lab5_results.png")
    print("=" * 70)

    return image, defect_mask, enhanced, S, D, N_blocks


# ==============================================================================
# ADDITIONAL ANALYSIS FOR REAL IMAGES
# ==============================================================================


def analyze_real_image(image_path):
    """Analyze real image for defects"""
    try:
        # Load image
        img = Image.open(image_path).convert("L")
        image = np.array(img, dtype=np.float64)

        print(f"\n{'='*70}")
        print(f"Analyzing real image: {image_path}")
        print(f"{'='*70}")
        print(f"Image shape: {image.shape}")

        # Detect defects
        defect_mask, enhanced, T = detect_defects(image)

        # Analyze blocks
        S, D, N_blocks = analyze_blocks(defect_mask)

        print(f"Defect pixels: {np.sum(defect_mask)}")
        print(f"Defective blocks: {N_blocks}")

        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(defect_mask, cmap="gray")
        axes[0, 1].set_title(f"Defect Mask (T={T:.2f})")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(enhanced, cmap="gray")
        axes[1, 0].set_title("Enhanced (Defects Highlighted)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(D, cmap="hot")
        axes[1, 1].set_title(f"Defective Blocks (N={N_blocks})")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig("lab5_real_image_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        return image, defect_mask, enhanced, N_blocks

    except FileNotFoundError:
        print(f"Image file '{image_path}' not found")
        return None, None, None, None


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

if __name__ == "__main__":
    import os

    os.chdir("/home/gaurav/Desktop/Class/Clg/Digital Image Processing/Lab/Lab5")

    print("Lab 5: Image Enhancement-III")
    print("Defect Detection and Block Analysis")
    print("=" * 70)

    # Part 1: Sample image analysis
    image, defect_mask, enhanced, S, D, N_blocks = part1_defect_detection()

    # Additional analysis for real images if available
    try:
        real_img, real_mask, real_enhanced, real_blocks = analyze_real_image(
            "night_road.jpg"
        )
    except:
        print("\nNote: Real image analysis skipped (file not available)")

    print("\n" + "=" * 70)
    print("LAB 5 COMPLETED SUCCESSFULLY")
    print("=" * 70)
