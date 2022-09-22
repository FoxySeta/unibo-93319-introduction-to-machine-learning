import numpy as np, cv2 as cv
from ml_intro import samples, utils


def to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def to_hsv(image):
    return cv.cvtColor(image, cv.COLOR_RGB2HSV)


def apply_mask(image, mask):
    return image * (np.equal(mask, 255)).astype(np.uint8).reshape(mask.shape + (1,))


def find_mask(image):
    _, red_threshold = cv.threshold(image[:, :, 0], 75, 255, cv.THRESH_BINARY)
    _, green_threshold_inv = cv.threshold(
        image[:, :, 1], 180, 255, cv.THRESH_BINARY_INV
    )
    _, blue_threshold = cv.threshold(image[:, :, 2], 75, 255, cv.THRESH_BINARY)
    hsv = to_hsv(image)
    _, hue_threshold = cv.threshold(image[:, :, 0], 90, 255, cv.THRESH_BINARY)
    _, hue_threshold_inv = cv.threshold(image[:, :, 0], 35, 255, cv.THRESH_BINARY_INV)
    return (
        (red_threshold | blue_threshold)
        & (hue_threshold | hue_threshold_inv)
        & green_threshold_inv
    )


def compute_difference_image(current_mask, target_mask):
    difference_image = np.zeros(current_mask.shape + (3,), dtype=np.uint8)
    boolean_current_mask = np.equal(current_mask, 255)
    boolean_target_mask = np.equal(target_mask, 255)
    difference_image[boolean_current_mask & boolean_target_mask] = (255, 255, 255)
    difference_image[~boolean_current_mask & boolean_target_mask] = (255, 0, 0)
    difference_image[boolean_current_mask & ~boolean_target_mask] = (0, 0, 255)
    return difference_image


def intersection_over_union(current_mask, target_mask):
    boolean_current_mask = np.equal(current_mask, 255)
    boolean_target_mask = np.equal(target_mask, 255)
    intersection = np.count_nonzero(boolean_current_mask & boolean_target_mask)
    union = np.count_nonzero(boolean_current_mask | boolean_target_mask)
    return intersection / union


original_parrots = samples.special.parrots
mask_parrots = find_mask(np.copy(original_parrots))
target_parrots = to_grayscale(samples.special.parrots_mask)
difference_image = compute_difference_image(mask_parrots, target_parrots)
utils.plot_images(
    [mask_parrots, target_parrots], space="gray", titles=["current mask", "target mask"]
)
utils.plot_images(
    [
        original_parrots,
        apply_mask(original_parrots, mask_parrots),
        apply_mask(original_parrots, target_parrots),
        difference_image,
    ],
    titles=["original", "masked", "target", "difference"],
    columns=4,
)
print(
    "Punteggio: {:.2f}%".format(
        intersection_over_union(mask_parrots, target_parrots) * 100
    )
)
