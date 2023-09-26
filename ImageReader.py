import cv2
import numpy as np
import math
import glob
## RGB = BGR

input_image = cv2.imread("CroppedDataset/1.jpg")

output = np.zeros((2, 5, 5), dtype=np.int8)
temp_output = np.zeros((input_image.shape[0],input_image.shape[1]), dtype=input_image.dtype)

for y, row in enumerate(input_image):
    for x, pixel in enumerate(row):
        newpixel = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3
        #print(type(newpixel))
        newpixel = int(newpixel)
        temp_output[y, x] = newpixel

imageHeight = input_image.shape[0]
imageWidth = input_image.shape[1]

M = int(imageHeight / 5)
N = int(imageWidth / 5)

for y in range(0, imageHeight, M):
    for x in range(0, imageWidth, N):
        tiles = np.array(input_image[y:y+M, x:x+N])



        cv2.imshow("hej", tiles)
         cv2.waitKey(0)
        #print(tiles)




temp_output = np.array_split(input_image, 5)
print(temp_output[0])


#cv2.imshow("image", input_image)
#cv2.waitKey(0)