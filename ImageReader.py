import cv2
import numpy as np
import math
import glob

input_image = cv2.imread("CroppedDataset/1.jpg")
output = np.zeros((2, 5, 5), dtype=np.int8)
temp_output = np.zeros((input_image.shape[0],input_image.shape[1]), dtype=input_image.dtype)

for y, row in enumerate(input_image):
    for x, pixel in enumerate(row):
        newpixel = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3
        newpixel = int(newpixel)
        temp_output[y, x] = newpixel

imageHeight = input_image.shape[0]
imageWidth = input_image.shape[1]

M = int(imageHeight / 5)
N = int(imageWidth / 5)

for y in range(0, imageHeight, M):
    for x in range(0, imageWidth, N):
        tiles = np.array(input_image[y:y + M, x:x + N])
        image_rgb = cv2.cvtColor(tiles, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape((-1, 3))
        k = 1
        # Perform k-means clustering to find the dominant color
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_colors = np.uint8(centers)
        print("Tile:", (int(x/100),int(y/100)))
        for color in dominant_colors:
            print("Dominant Color (RGB):", color, "\n")

        cv2.imshow("King Domino Board", tiles)
        cv2.waitKey(0)
        # print(tiles)

#temp_output = np.array_split(input_image, 5)
#print(temp_output[0])

#cv2.imshow("image", input_image)
#cv2.waitKey(0)