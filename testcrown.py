import cv2 as cv
import numpy as np

# Load the main image
img_rgb = cv.imread("CroppedDataset/10.jpg")
assert img_rgb is not None, "File could not be read, check with os.path.exists()"

# Convert the main image to grayscale
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

# Define a list of template file names
template_files = ['crowns/crown1.png', 'crowns/crown2.png', 'crowns/crown3.png', 'crowns/crown4.png']

# Loop through each template
for template_file in template_files:
    # Load the template
    template = cv.imread(template_file, cv.IMREAD_GRAYSCALE)
    assert template is not None, "File could not be read, check with os.path.exists()"

    # Perform template matching to find potential matches
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)

    # Create an empty mask to store the outlines
    mask = np.zeros_like(img_gray)

    for pt in zip(*loc[::-1]):
        x, y = pt
        w, h = template.shape[::-1]

        # Draw the outline of the matched area on the mask
        cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Find contours of the mask (outlines of matches)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original color image
    cv.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)

# Save the result image with outlines of matches
cv.imwrite('res.png', img_rgb)

# Display the result
cv.imshow("Result", img_rgb)
cv.waitKey(0)