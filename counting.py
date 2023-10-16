import numpy as np

# Define the 5x5x2 array with 7 different types (replace with your data)
array = np.array([[[3, 3, 1, 1, 5],
                   [3, 3, 3, 7, 5],
                   [4, 4, 3, 7, 6],
                   [4, 1, 1, 6, 7],
                   [1, 1, 1, 1, 7]],

                  [[0, 0, 2, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0],
                   [1, 0, 0, 0, 0]]])

def is_valid(x, y, z, layers, rows, cols):
    return 0 <= x < rows and 0 <= y < cols and 0 <= z < layers

def find_connected_blocks(z, x, y, target_type, layers, rows, cols, visited):
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
    queue = [(z, x, y)]
    connected_blocks = set()

    while queue:
        z, x, y = queue.pop()
        visited[z][x][y] = True
        connected_blocks.add((z, x, y))

        for i in range(4):
            new_x, new_y = x + dx[i], y + dy[i]
            if is_valid(new_x, new_y, z, layers, rows, cols) and not visited[z][new_x][new_y] and array[z][new_x][new_y] == target_type:
                queue.append((z, new_x, new_y))

    return connected_blocks

def analyze_blocks(array):
    layers, rows, cols = array.shape
    unique_values = np.unique(array)
    unique_values = unique_values[unique_values != 0]  # Exclude the background label (0)
    results = {}
    scores = {}

    visited = np.zeros((layers, rows, cols), dtype=bool)

    for x in range(rows):
        for y in range(cols):
            if array[0][x][y] != 0:  # Check if it's not the layer with 0s and 1s
                current_type = array[0][x][y]
                if current_type != 0 and not visited[0][x][y]:
                    connected_blocks = find_connected_blocks(0, x, y, current_type, layers, rows, cols, visited)
                    score = len(connected_blocks)  # Calculate the score based on the number of connected blocks

                    # Initialize the multiplier for this block
                    multiplier = 0

                    # Iterate through the connected blocks and add their multipliers
                    for block_z, block_x, block_y in connected_blocks:
                        multiplier += array[1][block_x][block_y]

                    # Multiply the score by the total multiplier
                    score *= multiplier

                    for block_z, block_x, block_y in connected_blocks:
                        visited[block_z][block_x][block_y] = True
                    results[current_type] = results.get(current_type, [])
                    scores[current_type] = scores.get(current_type, 0)
                    results[current_type].append((len(connected_blocks), connected_blocks))
                    scores[current_type] += score  # Add the score to the total score

    return results, scores

# Analyze the blocks and count the connected blocks for each type and calculate scores
results, scores = analyze_blocks(array)

# Print the results and scores
for label, block_info in results.items():
    for i, (block_size, connected_blocks) in enumerate(block_info):
        print(f"Block {i + 1} of Type {label} with {block_size} pixels is connected to:")
        for z, x, y in connected_blocks:
            print(f"  Pixel ({x}, {y}, {z}) of Type {label}")
        print(f"  Total Score for Type {label}: {scores[label]}")

# Calculate and print the total score for all types
total_score = sum(scores.values())
print(f"Total Score for all types: {total_score}")