import cv2
import numpy as np

# Load the image
image = cv2.imread("CroppedDataset/7.jpg")

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper HSV range for yellow (adjust V channel to exclude darker shades)
lower_yellow = np.array([25, 130, 130])
upper_yellow = np.array([35, 210, 200])

# Create a binary mask that isolates yellow regions
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Bitwise-AND the original image with the yellow mask
result = cv2.bitwise_and(image, image, mask=yellow_mask)

# Convert the result to grayscale
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Find contours in the image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store blobs that match the size criteria
filtered_blobs = []

# Define the size range and maximum width and height
min_area = 16 * 7
max_area = 20 * 16
max_width = 23
max_height = 23

# Iterate through the contours
for contour in contours:
    # Calculate the bounding rectangle for each contour
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h

    # Check if the blob's area is within the specified range and if its width and height are within the maximum limits
    if min_area <= area <= max_area and w <= max_width and h <= max_height:
        filtered_blobs.append((x, y, w, h))

# Determine the grid size
grid_size = 5

# Initialize a numpy array to store the counts of blobs in each grid cell
grid_counts = np.zeros((grid_size, grid_size), dtype=np.int32)

# Calculate the size of each grid cell
cell_height = image.shape[0] // grid_size
cell_width = image.shape[1] // grid_size

# Iterate through the filtered blobs and count them in each grid cell
for x, y, w, h in filtered_blobs:
    cell_x = x // cell_width
    cell_y = y // cell_height
    grid_counts[cell_y, cell_x] += 1

# Display the counts
print("Grid Counts:")
print(grid_counts)

result_with_blobs = image.copy()
for x, y, w, h in filtered_blobs:
    cv2.rectangle(result_with_blobs, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color, thickness = 2

# Display the result (image with yellow and blob counts)
cv2.imshow('Yellow Objects with Blobs', result_with_blobs)
cv2.imshow('Yellow Objects', result)

cv2.waitKey(0)
cv2.destroyAllWindows()