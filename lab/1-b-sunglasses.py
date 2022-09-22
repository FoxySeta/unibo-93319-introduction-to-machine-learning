import numpy as np, cv2 as cv
from ml_intro import samples, utils


def draw_sunglasses(image, x, y, width, height):
    sunglasses_alpha = cv.resize(
        np.copy(samples.special.sunglasses_alpha), [width, height]
    )
    actual_width = max(min(x + width, image.shape[1]) - x, 0)
    actual_height = max(min(y + height, image.shape[0]) - y, 0)
    if actual_width == 0 or actual_height == 0:
        return image
    sunglasses_alpha = sunglasses_alpha[:actual_height, :actual_width, :]
    image = np.copy(image)
    mask = sunglasses_alpha[:, :, 3] / 255
    mask = mask.reshape([sunglasses_alpha.shape[0], sunglasses_alpha.shape[1], 1])
    image[y : y + actual_height, x : x + actual_width, :3] = sunglasses_alpha[
        :, :, :3
    ] * mask + image[y : y + actual_height, x : x + actual_width, :3] * (1 - mask)
    return image


def to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def black_bounding_box(image):
    grayscale = to_grayscale(image)
    _, black = cv.threshold(grayscale[:, :], 15, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    x, y, width, height = cv.boundingRect(contours[0])
    x2, y2, width2, height2 = cv.boundingRect(contours[1])
    return (
        min(x, x2),
        min(y, y2),
        max(x + width, x2 + width2),
        max(y + height, y2 + height2),
    )


def sunglasses_on_face(image):
    x, y, x2, _ = black_bounding_box(image)
    width = x2 - x
    sunglasses_shape = samples.special.sunglasses_alpha.shape
    height = width * sunglasses_shape[0] // sunglasses_shape[1]
    return draw_sunglasses(image, x, y, width, height)


utils.plot_images([samples.bird, sunglasses_on_face(samples.bird)])
