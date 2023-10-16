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
            print("\nTile:", (int(x / 100), int(y / 100)))
            for color in dominant_colors:
                print("Dominant Color (BGR):", color)

                print(determineColor(color))

            cv2.imshow("King Domino Board", tiles)
            cv2.waitKey(0)
            # print(tiles)

def determineColor(color): # could use a for-loop :)
    print("Biome: ", end="")
    for i in biome_dict:
        watched_biome = biome_dict[i]

        for j in enumerate(bgr_dict):
            bgr_dict.update({j[1]: False})
            if watched_biome[0][j[0]] <= color[j[0]] <= watched_biome[1][j[0]]:
                bgr_dict.update({j[1]: True})

        if bgr_dict.get("blue") is True and bgr_dict.get("green") is True and bgr_dict.get("red") == True:
            return str(i)
    else:
        return "castle"


biome_dict = {
    "forest":  [[15, 29, 10], [67, 91, 65]],
    "plains":  [[103, 69, 1], [201, 175, 24]],
    "grass":   [[72, 110, 0], [130, 162, 40]],
    "waste":    [[69, 46, 12], [141, 132, 101]],
    "ocean":   [[4, 44, 90], [58, 112, 211]],
    "mine":    [[21, 21, 11], [158, 128, 38]]
}

bgr_dict = {
    "blue": False,
    "green": False,
    "red": False

}

findDominantRGB()

#temp_output = np.array_split(input_image, 5)
#print(temp_output[0])
        tiles = np.array(input_image[y:y+M, x:x+N])



        cv2.imshow("hej", tiles)
        cv2.waitKey(0)
        #print(tiles)




temp_output = np.array_split(input_image, 5)
print(temp_output[0])


#cv2.imshow("image", input_image)
#cv2.waitKey(0)