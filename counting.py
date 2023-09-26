import cv2
import numpy as np

lion_image = cv2.imread("lion.jpg")
output_image = np.zeros((lion_image.shape[0],lion_image.shape[1]), dtype=lion_image.dtype)

for y, row in enumerate(lion_image):
    for x, pixel in enumerate(row):
        newpixel = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3
        print(type(newpixel))
        newpixel = int(newpixel)
        output_image[y, x] = newpixel

cv2.imshow("Our window",lion_image)
cv2.imshow("ok",output_image)
cv2.waitKey(0)