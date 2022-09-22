import numpy as np, cv2 as cv
from ml_intro import samples, utils

bin_length = 50


def posterize(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    image[:, :, 2] //= bin_length
    image[:, :, 2] *= bin_length
    return cv.cvtColor(image, cv.COLOR_HSV2RGB)


utils.plot_images(
    [samples.bird, posterize(samples.bird)], titles=["original", "posterized"]
)
