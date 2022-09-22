from math import ceil

import cv2 as cv
import matplotlib.pyplot as plt

def normalize_image(image):
    pass

def plot_image(image, title=None, space='RGB'):
    plot_images([image], titles=(None if title is None else [title]), space=space)

def plot_images(images, titles=None, columns=2, space='RGB'):
    columns = min(columns, len(images))
    rows = int(ceil(len(images) / columns))

    if space.lower() == 'hsv':
        images = [cv.cvtColor(image, cv.COLOR_HSV2RGB) for image in images]
    elif space.lower() == 'gray':
        images = [cv.cvtColor(image, cv.COLOR_GRAY2RGB) for image in images]
    elif space.lower() != 'rgb':
        raise ValueError(f'Unsupported color space {space}. Allowed values are RGB, Gray and HSV.')

    if columns == 1 and rows == 1:
        plt.imshow(images[0])
        if titles is not None:
            plt.title(titles[0])
        plt.axis('off')
    else:
        _, axs = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
        axs = axs.flatten()

        for i in range(len(images)):
            axs[i].imshow(images[i])
            if titles is not None:
                axs[i].set_title(titles[i])
        for axis in axs:
            axis.axis('off')
    plt.show()
    #plt.show()