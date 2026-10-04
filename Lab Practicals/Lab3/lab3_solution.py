"""
Lab 3: Pixel Geometry
Digital Image Processing (NCSH302)

This solution implements:
1. Pixel adjacency (4-adjacency, 8-adjacency, m-adjacency)
2. Path finding between two pixels
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# ==============================================================================
# PART 1: Pixel Adjacency Functions
# ==============================================================================

def get_neighbors_4(image, x, y):
    """Get 4-neighbors of pixel at (x, y)"""
    rows, cols = image.shape
    neighbors = []
    
    # 4-neighbors positions: (x-1,y), (x+1,y), (x,y-1), (x,y+1)
    positions = [
        (x - 1, y),  # up
        (x + 1, y),  # down
        (x, y - 1),  # left
        (x, y + 1)   # right
    ]
    
    for nx, ny in positions:
        if 0 <= nx < rows and 0 <= ny < cols:
            neighbors.append(((nx, ny), image[nx, ny]))
    
    return neighbors

def get_neighbors_diagonal(image, x, y):
    """Get diagonal neighbors of pixel at (x, y)"""
    rows, cols = image.shape
    neighbors = []
    
    # Diagonal neighbors
    positions = [
        (x - 1, y - 1),  # top-left
        (x - 1, y + 1),  # top-right
        (x + 1, y - 1),  # bottom-left
        (x + 1, y + 1)   # bottom-right
    ]
    
    for nx, ny in positions:
        if 0 <= nx < rows and 0 <= ny < cols:
            neighbors.append(((nx, ny), image[nx, ny]))
    
    return neighbors

def get_neighbors_8(image, x, y):
    """Get 8-neighbors of pixel at (x, y)"""
    n4 = get_neighbors_4(image, x, y)
    nd = get_neighbors_diagonal(image, x, y)
    return n4 + nd

def get_neighbors_m(image, x, y, V):
    """
    Get m-neighbors of pixel at (x, y)
    m-adjacency: q is m-adjacent to p if:
    - q is 4-adjacent to p, OR
    - q is diagonal-adjacent to p AND N4(p) ∩ N4(q) has no pixels with values in V
    """
    neighbors = []
    rows, cols = image.shape
    
    # Add all 4-neighbors
    n4_neighbors = get_neighbors_4(image, x, y)
    for (nx, ny), val in n4_neighbors:
        neighbors.append(((nx, ny), val))
    
    # Add diagonal neighbors if condition is satisfied
    diagonal_neighbors = get_neighbors_diagonal(image, x, y)
    for (nx, ny), val in diagonal_neighbors:
        # Check if N4(p) ∩ N4(q) has no pixels with values in V
        n4_p = get_neighbors_4(image, x, y)
        n4_q = get_neighbors_4(image, nx, ny)
        
        # Get intersection positions
        n4_p_set = {pos for pos, _ in n4_p}
        n4_q_set = {pos for pos, _ in n4_q}
        intersection = n4_p_set & n4_q_set
        
        # Check if intersection has any pixels in V
        has_V = False
        for ipos in intersection:
            if image[ipos[0], ipos[1]] in V:
                has_V = True
                break
        
        # Add diagonal neighbor if no V pixels in intersection
        if not has_V:
            neighbors.append(((nx, ny), val))
    
    return neighbors

def pixel_adj(image, adj_type, p, V):
    """
    Find adjacent pixels
    adj_type: '4', '8', or 'm'
    p: pixel position (x, y)
    V: set of intensity values to match
    """
    x, y = p
    
    if adj_type == '4':
        neighbors = get_neighbors_4(image, x, y)
    elif adj_type == '8':
        neighbors = get_neighbors_8(image, x, y)
    elif adj_type == 'm':
        neighbors = get_neighbors_m(image, x, y, V)
    else:
        raise ValueError("adj_type must be '4', '8', or 'm'")
    
    # Filter by value set V
    adjacent = [((nx, ny), val) for (nx, ny), val in neighbors if val in V]
    
    return adjacent

def part1_pixel_adjacency():
    """Part 1: Demonstrate pixel adjacency operations"""
    print("=" * 70)
    print("PART 1: Pixel Adjacency")
    print("=" * 70)
    
    # Create sample image (8x8)
    image = np.array([
        [3, 5, 3, 5, 2, 4, 3, 2],
        [4, 6, 6, 7, 3, 3, 7, 6],
        [6, 3, 0, 3, 1, 3, 2, 3],
        [5, 4, 2, 1, 2, 3, 6, 7],
        [2, 7, 3, 0, 0, 1, 2, 5],
        [7, 4, 1, 0, 2, 3, 3, 4],
        [4, 2, 7, 3, 5, 7, 2, 5],
        [2, 6, 5, 4, 2, 7, 5, 6]
    ], dtype=np.uint8)
    
    print(f"Image shape: {image.shape}")
    print("\nImage:")
    print(image)
    
    # Test pixel
    p = (3, 3)  # 1-based would be (4, 4), but we use 0-based indexing
    V = {0, 1}  # Set of values to find
    
    print(f"\nPixel position p = {p}")
    print(f"Pixel value at p: {image[p[0], p[1]]}")
    print(f"Value set V = {V}")
    
    # 4-adjacency
    print("\n4-adjacency:")
    adj_4 = pixel_adj(image, '4', p, V)
    for (nx, ny), val in adj_4:
        print(f"  Value = {val}, position = ({nx}, {ny})")
    
    # 8-adjacency
    print("\n8-adjacency:")
    adj_8 = pixel_adj(image, '8', p, V)
    for (nx, ny), val in adj_8:
        print(f"  Value = {val}, position = ({nx}, {ny})")
    
    # m-adjacency
    print("\nm-adjacency:")
    adj_m = pixel_adj(image, 'm', p, V)
    for (nx, ny), val in adj_m:
        print(f"  Value = {val}, position = ({nx}, {ny})")
    
    return image, p, V


# ==============================================================================
# PART 2: Path Finding
# ==============================================================================

def find_all_paths_4(image, start, end, V, max_paths=100):
    """Find all 4-paths between start and end pixels"""
    paths = []
    queue = deque([(start, [start])])
    visited_per_path = set()
    
    while queue and len(paths) < max_paths:
        (x, y), path = queue.popleft()
        
        if (x, y) == end:
            paths.append(path)
            continue
        
        # Get 4-neighbors
        neighbors = get_neighbors_4(image, x, y)
        for (nx, ny), val in neighbors:
            if val in V and (nx, ny) not in path:
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return paths

def find_all_paths_8(image, start, end, V, max_paths=100):
    """Find all 8-paths between start and end pixels"""
    paths = []
    queue = deque([(start, [start])])
    
    while queue and len(paths) < max_paths:
        (x, y), path = queue.popleft()
        
        if (x, y) == end:
            paths.append(path)
            continue
        
        # Get 8-neighbors
        neighbors = get_neighbors_8(image, x, y)
        for (nx, ny), val in neighbors:
            if val in V and (nx, ny) not in path:
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return paths

def find_all_paths_m(image, start, end, V, max_paths=100):
    """Find all m-paths between start and end pixels"""
    paths = []
    queue = deque([(start, [start])])
    
    while queue and len(paths) < max_paths:
        (x, y), path = queue.popleft()
        
        if (x, y) == end:
            paths.append(path)
            continue
        
        # Get m-neighbors
        neighbors = get_neighbors_m(image, x, y, V)
        for (nx, ny), val in neighbors:
            if val in V and (nx, ny) not in path:
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return paths

def part2_path_finding():
    """Part 2: Find paths between two pixels"""
    print("\n" + "=" * 70)
    print("PART 2: Path Finding")
    print("=" * 70)
    
    # Create sample 7x7 image for path finding
    image = np.array([
        [3, 0, 3, 5, 2, 4, 3],
        [4, 6, 1, 7, 3, 3, 7],
        [6, 3, 0, 3, 1, 3, 2],
        [5, 4, 2, 1, 2, 3, 6],
        [2, 7, 3, 0, 0, 1, 2],
        [7, 4, 1, 0, 2, 3, 3],
        [4, 2, 7, 3, 5, 7, 2]
    ], dtype=np.uint8)
    
    print(f"Image shape: {image.shape}")
    print("\nImage:")
    print(image)
    
    # Define start and end points
    start = (0, 0)  # Top-left
    end = (6, 6)    # Bottom-right
    V = {0, 1}      # Pixels with values 0 or 1
    
    print(f"\nStart point: {start}, value = {image[start[0], start[1]]}")
    print(f"End point: {end}, value = {image[end[0], end[1]]}")
    print(f"Value set V = {V}")
    
    # Find 4-paths
    print("\nFinding 4-paths...")
    paths_4 = find_all_paths_4(image, start, end, V, max_paths=10)
    print(f"Found {len(paths_4)} 4-paths")
    if paths_4:
        print(f"First 4-path (length {len(paths_4[0])}): {paths_4[0]}")
    
    # Find 8-paths
    print("\nFinding 8-paths...")
    paths_8 = find_all_paths_8(image, start, end, V, max_paths=10)
    print(f"Found {len(paths_8)} 8-paths")
    if paths_8:
        print(f"First 8-path (length {len(paths_8[0])}): {paths_8[0]}")
    
    # Find m-paths
    print("\nFinding m-paths...")
    paths_m = find_all_paths_m(image, start, end, V, max_paths=10)
    print(f"Found {len(paths_m)} m-paths")
    if paths_m:
        print(f"First m-path (length {len(paths_m[0])}): {paths_m[0]}")
    
    return image, start, end, V, paths_4, paths_8, paths_m


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_paths(image, paths, title):
    """Visualize paths on image"""
    vis = np.zeros((*image.shape, 3), dtype=np.uint8)
    
    # Normalize image to 0-255 for visualization
    img_normalized = ((image / image.max()) * 255).astype(np.uint8)
    vis[:, :, 0] = img_normalized
    vis[:, :, 1] = img_normalized
    vis[:, :, 2] = img_normalized
    
    # Draw paths
    for path_idx, path in enumerate(paths[:5]):  # Draw first 5 paths
        color = [np.random.randint(100, 255) for _ in range(3)]
        for x, y in path:
            vis[x, y] = color
    
    return vis

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("Lab 3: Pixel Geometry")
    print("=" * 70)
    
    # Part 1
    image1, p1, V1 = part1_pixel_adjacency()
    
    # Part 2
    image2, start, end, V2, paths_4, paths_8, paths_m = part2_path_finding()
    
    # Visualize
    if paths_4:
        vis_4 = visualize_paths(image2, paths_4, "4-paths")
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_4)
        plt.title("Visualization of 4-paths")
        plt.savefig('lab3_paths_4.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n" + "=" * 70)
    print("LAB 3 COMPLETED SUCCESSFULLY")
    print("=" * 70)
