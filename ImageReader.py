import cv2
import numpy as np

input_image = cv2.imread("CroppedDataset/7.jpg")
output = np.zeros((2, 5, 5), dtype=np.int8)
temp_output = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=input_image.dtype)
biome_array = np.zeros((2, 5, 5), dtype=np.int8)

for y, row in enumerate(input_image):
    for x, pixel in enumerate(row):
        new_pixel = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3
        new_pixel = int(new_pixel)
        temp_output[y, x] = new_pixel

imageHeight = input_image.shape[0]
imageWidth = input_image.shape[1]

M = int(imageHeight / 5)
N = int(imageWidth / 5)


def find_dominant_rgb():
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
                print("Biome:", determine_biome(color))
                biome_array[0][int(y / 100)][int(x / 100)] = determine_biome(color)

            cv2.imshow("King Domino Board", tiles)
            cv2.waitKey(0)
            # print(tiles)
    print(biome_array)


def determine_biome(color):
    #print("Biome: ", end="")
    biome_index = 0
    #biome_bool = np.array(6*[False], dtype=np.bool_)
    biome_bool = len(biome_dict)*[False]
    for i in biome_dict:
        watched_biome = biome_dict[i]

        for j in enumerate(bgr_dict):
            bgr_dict.update({j[1]: False})
            if watched_biome[0][j[0]] <= color[j[0]] <= watched_biome[1][j[0]]:
                bgr_dict.update({j[1]: True})

        if bgr_dict.get("blue") is True and bgr_dict.get("green") is True and bgr_dict.get("red") is True:
            biome_bool[biome_index] = True
        biome_index += 1

    #this, for some reason, always returns 6, even though the array is correct
    if biome_bool[0] and biome_bool[5]:
        biome_index = list(biome_dict.keys()).index(forestOrMine(color))
        return biome_index
    else:
        for i in range(len(biome_bool)):
            if biome_bool[i]:
                return i
        #if there is no matches
        return 6

def forestOrMine(color):
    forest_difference = [0, 0, 0]
    mine_difference = [0, 0, 0]
    for i in range(len(color)):
        forest_difference[i] = abs((biome_dict["forest"][1][i] - biome_dict["forest"][0][i]) - color[i])
        mine_difference[i] = abs((biome_dict["mine"][1][i] - biome_dict["mine"][0][i]) - color[i])
    #this is not the best way to do it, as having two lower entries always "wins"
    if forest_difference > mine_difference: #switch </> if results are wonky
        return "forest"
    else:
        return "mine"


biome_dict = {
    "forest":   [[29, 41, 11], [66, 72, 41]],
    "plains":   [[103, 69, 1], [201, 175, 24]],
    "grass":    [[72, 110, 0], [130, 162, 40]],
    "waste":    [[69, 46, 12], [141, 132, 101]],
    "ocean":    [[2, 44, 90], [58, 112, 211]],
    "mine":     [[50, 41, 19], [78, 66, 35]]
}

bgr_dict = {
    "blue": False,
    "green": False,
    "red": False
}

find_dominant_rgb()



# dunno if this was used for anything or just for testing?
#cv2.imshow("hej", tiles)
#cv2.waitKey(0)
#print(tiles)

temp_output = np.array_split(input_image, 5)
print(temp_output[0])
#tiles = np.array(input_image[y:y+M, x:x+N])
