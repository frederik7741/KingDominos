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

def findDominantRGB():
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
            print("Tile:", (int(x / 100), int(y / 100)))
            for color in dominant_colors:
                print("Dominant Color (BGR):", color, "\n")

                determineColor(color)

            cv2.imshow("King Domino Board", tiles)
            cv2.waitKey(0)
            # print(tiles)

def determineColor(color): # needs fixing
    if (forest_low_range[0] <= color[0] <= forest_up_range[0] and
        forest_low_range[1] <= color[1] <= forest_up_range[1] and
        forest_low_range[2] <= color[2] <= forest_up_range[2]):
        print("forest")
    elif (plains_low_range[0] <= color[0] <= plains_up_range[0] and
        plains_low_range[1] <= color[1] <= plains_up_range[1] and
        plains_low_range[2] <= color[2] <= plains_up_range[2]):
        print("plains")
    elif (grass_low_range[0] <= color[0] <= grass_up_range[0] and
        grass_low_range[1] <= color[1] <= grass_up_range[1] and
        grass_low_range[2] <= color[2] <= grass_up_range[2]):
        print("plains")
    elif (waste_up_range[0] <= color[0] <= waste_low_range[0] and
        waste_up_range[1] <= color[1] <= waste_low_range[1] and
        waste_up_range[2] <= color[2] <= waste_low_range[2]):
        print("plains")
    elif (plains_low_range[0] <= color[0] <= plains_up_range[0] and
        plains_low_range[1] <= color[1] <= plains_up_range[1] and
        plains_low_range[2] <= color[2] <= plains_up_range[2]):
        print("plains")
    elif (plains_low_range[0] <= color[0] <= plains_up_range[0] and
        plains_low_range[1] <= color[1] <= plains_up_range[1] and
        plains_low_range[2] <= color[2] <= plains_up_range[2]):
        print("plains")
    else:
        print("castle")


up_range = [255, 255, 255]
low_range = [0, 0, 0]

# Forest
forest_up_range = [46, 91, 65]
forest_low_range = [15, 29, 16]
# Plains
plains_up_range = [103, 69, 8]
plains_low_range = [198, 172, 15]
# Grasslands
grass_up_range = [72, 123, 0]
grass_low_range = [130, 162, 32]
# Wasteland
waste_up_range = [141, 132, 101]
waste_low_range = [69, 46, 12]
# Ocean
ocean_up_range = [8, 112, 211]
ocean_low_range = [19, 58, 113]
# Mine
mine_up_range = [158, 128, 38]
mine_low_range = [21, 21, 11]


findDominantRGB()

#temp_output = np.array_split(input_image, 5)
#print(temp_output[0])

#cv2.imshow("image", input_image)
#cv2.waitKey(0)