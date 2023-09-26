import cv2
import numpy as np
import math

# Define the 5x5 array with 7 different types (replace with your data)
array = np.array([[1, 1, 2, 2, 2],
                  [1, 1, 2, 2, 3],
                  [4, 4, 5, 3, 3],
                  [4, 5, 2, 2, 6],
                  [7, 7, 6, 6, 6]])

def is_valid(x, y, rows, cols):
    return 0 <= x < rows and 0 <= y < cols

def find_connected_blocks(x, y, target_type, rows, cols, visited):
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
    queue = [(x, y)]
    connected_blocks = set()

    while queue:
        x, y = queue.pop()
        visited[x][y] = True
        connected_blocks.add((x, y))

        for i in range(4):
            new_x, new_y = x + dx[i], y + dy[i]
            if is_valid(new_x, new_y, rows, cols) and not visited[new_x][new_y] and array[new_x][new_y] == target_type:
                queue.append((new_x, new_y))

    return connected_blocks

def analyze_blocks(array):
    rows, cols = array.shape
    unique_values = np.unique(array)
    unique_values = unique_values[unique_values != 0]  # Exclude the background label (0)
    results = {}

    visited = np.zeros((rows, cols), dtype=bool)

    for x in range(rows):
        for y in range(cols):
            if not visited[x][y]:
                current_type = array[x][y]
                if current_type != 0:
                    connected_blocks = find_connected_blocks(x, y, current_type, rows, cols, visited)
                    for block_x, block_y in connected_blocks:
                        visited[block_x][block_y] = True
                    results[current_type] = results.get(current_type, [])
                    results[current_type].append((len(connected_blocks), connected_blocks))

    return results

# Analyze the blocks and count the connected blocks for each type
results = analyze_blocks(array)

# Print the results
for label, block_info in results.items():
    for i, (block_size, connected_blocks) in enumerate(block_info):
        print(f"Block {i + 1} of Type {label} with {block_size} pixels is connected to:")
        for x, y in connected_blocks:
            print(f"  Pixel ({x}, {y}) of Type {label}")