import numpy as np
import cv2
from choose_image import get_image_index

image_number = get_image_index()
# Load the image
image = cv2.imread(f"CroppedDataset/{image_number}.jpg")

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper HSV range for yellow (adjust V channel to exclude darker shades)
lower_yellow = np.array([25, 130, 130])
upper_yellow = np.array([35, 210, 200])

# Create a binary mask that isolates yellow regions
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Display the yellow mask before closing
cv2.imshow('Yellow Mask (Before Closing)', yellow_mask)
cv2.waitKey(0)

# Apply closing to the mask
kernel1 = np.ones((5, 5), np.uint8)
kernel2 = np.ones((3, 3), np.uint8)

yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel1, iterations=1)

# Display the yellow mask after closing
cv2.imshow('Yellow Mask (After Closing)', yellow_mask)
cv2.waitKey(0)

# Apply erosion to the mask after closing
yellow_mask = cv2.erode(yellow_mask, kernel2, iterations=1)

# Display the yellow mask after erosion
cv2.imshow('Yellow Mask (After Erosion)', yellow_mask)
cv2.waitKey(0)

# Find contours in the binary mask (yellow_mask)
contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define the minimum and maximum area for a blob (in pixels)
min_blob_area = 30  # Adjust this value
max_blob_area = 120  # Adjust this value

# Initialize a list to store pruned blobs
pruned_blobs = []

# Iterate through the contours and prune them based on area
for contour in contours:
    area = cv2.contourArea(contour)

    if min_blob_area <= area <= max_blob_area:
        pruned_blobs.append(contour)

# Create a new binary mask with pruned blobs
pruned_mask = np.zeros_like(yellow_mask)
cv2.drawContours(pruned_mask, pruned_blobs, -1, 255, thickness=cv2.FILLED)

# Save the pruned mask as an image
cv2.imshow('PrunedMask.jpg', pruned_mask)

# Create a 5x5 array to store pruned blobs for each block
block_blobs = [[[] for _ in range(5)] for _ in range(5)]

# Determine the width and height of each block in the image
block_width = image.shape[1] // 5
block_height = image.shape[0] // 5

# Define the size of the middle region to exclude (50 pixels from each side)
middle_size = 50

# Define the size of the border region to exclude (5 pixels from each side)
border_size = 3

# Iterate through the pruned blobs and add them to the block_blobs array
for contour in pruned_blobs:
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the block position
    block_x = min(x // block_width, 4)
    block_y = min(y // block_height, 4)

    # Check if the blob falls within the middle region of the block
    if (
        x + w >= block_x * block_width + middle_size
        and x <= (block_x + 1) * block_width - middle_size
        and y + h >= block_y * block_height + middle_size
        and y <= (block_y + 1) * block_height - middle_size
    ):
        continue  # Skip blobs in the middle region

    # Check if the blob falls within the border region of the block
    if (
        x <= block_x * block_width + border_size
        or x + w >= (block_x + 1) * block_width - border_size
        or y <= block_y * block_height + border_size
        or y + h >= (block_y + 1) * block_height - border_size
    ):
        continue  # Skip blobs in the border region

    # Add the pruned blob to the corresponding block
    block_blobs[block_y][block_x].append(contour)

# Draw red boxes around the pruned blobs on the original image
for i in range(5):
    for j in range(5):
        for contour in block_blobs[i][j]:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the result (image with red boxes around pruned blobs)
cv2.imshow('Yellow Objects with Pruned Blobs', image)

# Iterate through the 5x5 block_blobs array and count the number of contours in each block
block_blob_counts = [[len(block_blobs[i][j]) for j in range(5)] for i in range(5)]

# Print the resulting count array
crown_array = np.zeros((5, 5), dtype=np.int8)
print("Block Blob Counts:")
for i, row in enumerate(block_blob_counts):
    crown_array[i] = row

def get_crowns():
    return crown_array


# Release OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()