import numpy as np, cv2 as cv
from ml_intro import samples, utils

def to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

def dilate(image, dilation_size):
    dilation_shape = cv.MORPH_ELLIPSE
    dilation_element = cv.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))
    return cv.dilate(image, dilation_element)


gray_people = to_grayscale(samples.special.people)
_, people_mask = cv.threshold(gray_people, 8, 255, cv.THRESH_BINARY_INV)
people_mask = dilate(people_mask, 3)
contours, _ = cv.findContours(people_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
image_with_contours = cv.drawContours(samples.special.people.copy(), contours, -1, (255, 0, 0), 3)
utils.plot_image(image_with_contours)
print('Number of people:', len(contours))
