import numpy as np, cv2 as cv
from ml_intro import samples, utils


def apply_filter(image):
    image = image.copy()
    image[:, :, 0] += 3  # Red
    image[:, :, 1] -= 18 # Green
    image[:, :, 2] -= 11  # Blue
    image = cv.cvtColor(
        np.clip(image, 0, 255).astype(np.uint8), cv.COLOR_RGB2HSV
    ).astype(np.uint16)
    image[:, :, 0] += 0  # Hue
    image[:, :, 1] -= 43 # Saturation
    image[:, :, 2] += 2  # Value
    return cv.cvtColor(
        np.clip(image, 0, 255).astype(np.uint8), cv.COLOR_HSV2RGB
    )


reference_image = samples.filter.ludwig
transformed_image = apply_filter(samples.balloon)
print("Distance:", np.mean((reference_image - transformed_image) ** 2))
utils.plot_images(
    [samples.balloon, reference_image, transformed_image],
    titles=["Original", "Target", "Result"],
    columns=3,
)
