import numpy as np, cv2 as cv
from ml_intro import samples, utils


def smart_fusion(base_image, rgba_image):
    alpha_channel = rgba_image[:, :, 3] / 255
    rgb_channels = rgba_image[:, :, :3]
    alpha_channel = alpha_channel.reshape(
        alpha_channel.shape[0], alpha_channel.shape[1], 1
    )
    composite = rgb_channels * alpha_channel + base_image * (1 - alpha_channel)
    return composite.astype(np.uint8)


def to_hsv(image):
    return cv.cvtColor(image, cv.COLOR_RGB2HSV)


def to_rgb(image):
    return cv.cvtColor(image, cv.COLOR_HSV2RGB)


def tint(image, red_diff, green_diff, blue_diff):
    image = np.copy(image).astype(np.int16)
    image[:, :, 0] += red_diff
    image[:, :, 1] += green_diff
    image[:, :, 2] += blue_diff
    return np.clip(image, 0, 255).astype(np.uint8)


def build_rgba(base_image, mask):
    rgba_image = np.zeros([base_image.shape[0], base_image.shape[1], 4])
    rgba_image[:, :, :3] = base_image
    rgba_image[:, :, 3] = mask
    return rgba_image


def shining():
    image_shape = [samples.shining.duvall.shape[0], samples.shining.duvall.shape[1]]

    # Partiamo con il muro
    wall_red = tint(samples.shining.wall, 0, -170, -170)

    # Aggiungiamoci nicholson usando la maschera del coltello
    nicholson_blue = tint(samples.shining.nicholson, -20, -50, 0)
    nicholson_blue_hsv = to_hsv(nicholson_blue)
    nicholson_blue_hsv[:, :, 1] -= 60
    nicholson_blue_hsv[:, :, 2] += 20
    nicholson_blue = to_rgb(nicholson_blue_hsv)
    knife_mask = (samples.shining.knife_mask[:, :, 0] > 128).astype(
        np.uint8
    ) * 255  # Avrei potuto usare qualunque altro canale
    nicholson_rgba = build_rgba(nicholson_blue, knife_mask)
    fused = smart_fusion(wall_red, nicholson_rgba)

    # Facciamo duvall con tinta blu
    duvall_blue = tint(samples.shining.duvall, -50, -20, 0)
    # Esercizio extra: come avremmo potuto ottenere un effetto "coltello aggiuntivo"?

    # Fondiamo con la maschera di duvall
    duvall_alpha = np.logical_or(
        np.logical_or(
            samples.shining.duvall[:, :, 0] < 250, samples.shining.duvall[:, :, 1] < 250
        ),
        samples.shining.duvall[:, :, 2] < 250,
    )
    duvall_alpha = duvall_alpha.astype(np.uint8) * 255
    duvall_rgba = np.zeros([image_shape[0], image_shape[1], 4])
    duvall_rgba[:, :, :3] = duvall_blue
    duvall_rgba[:, :, 3] = duvall_alpha
    fused = smart_fusion(fused, duvall_rgba)

    # Aggiungiamo il titolo
    title_alpha = samples.shining.title[:, :, 0] > 200
    title_alpha = title_alpha.astype(np.uint8) * 255
    title_rgba = np.zeros([image_shape[0], image_shape[1], 4])
    title_rgba[:, :, :3] = samples.shining.title
    title_rgba[:, :, 3] = title_alpha

    fused = smart_fusion(fused, title_rgba)

    # E, infine, il testo
    fused = cv.putText(
        fused,
        "JACK NICHOLSON",
        (100, 100),
        cv.FONT_HERSHEY_DUPLEX,
        1.85,
        [255, 255, 255],
        5,
    )
    fused = cv.putText(
        fused,
        "SHELLEY DUVAL",
        (750, 100),
        cv.FONT_HERSHEY_DUPLEX,
        1.85,
        [255, 255, 255],
        5,
    )

    return fused


utils.plot_image(shining())
