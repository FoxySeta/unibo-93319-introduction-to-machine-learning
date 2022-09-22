from PIL import Image
import os
from pathlib import Path

import numpy as np

def load_sample(name, rescale=False, drop_alpha=True):
    base_path = os.path.abspath(os.path.dirname(__file__))
    path = Path(base_path) / 'img' / (name + '.png')
    image = Image.open(path)
    if drop_alpha:
        image = image.convert('RGB')
    image = np.asarray(image)

    if rescale:
        image = image.astype(np.float) / 255
    return image

image_names = ['balloon', 'bird', 'fireworks', 'fruit', 'mario', 'napoleon', 'newspaper', 'newyork', 'plane', 'scientist', 'statue', 'tuscany']
standard_images = []

for image_name in image_names:
    image = load_sample(image_name)
    standard_images.append(image)
    globals()[image_name] = image

class Special:
    def __init__(self):
        self.sunglasses_alpha = load_sample('special/sunglasses_alpha', drop_alpha=False)
        self.sunglasses_noalpha = load_sample('special/sunglasses_noalpha')
        self.bw_gradient = load_sample('special/bw_gradient')
        self.apple = load_sample('special/apple')
        self.sudoku = load_sample('special/sudoku')
        self.boomer = load_sample('special/boomer')
        self.parrots = load_sample('special/parrots')
        self.parrots_mask = load_sample('special/parrots_mask')
        self.birds = load_sample('special/birds')
        self.coins = load_sample('special/coins')
        self.people = load_sample('special/people')

special = Special()

class Filter:
    def __init__(self) -> None:
        self.names = ['aden', 'clarendon', 'crema', 'gingham', 'juno', 'lark', 'ludwig', 'moon', 'perpetua', 'reyes', 'slumber']
        self.images = [load_sample(f'filters/{name}') for name in self.names]
        for name, image in zip(self.names, self.images):
            setattr(self, name, image)

filter = Filter()

class Shining:
    def __init__(self) -> None:
        self.duvall = load_sample('shining/duvall')
        self.knife_mask = load_sample('shining/knife_mask')
        self.nicholson = load_sample('shining/nicholson')
        self.title = load_sample('shining/title')
        self.wall = load_sample('shining/wall')

shining = Shining()

class Histogram:
    def __init__(self) -> None:
        self.apples = [load_sample(f'histogram/apple_{i}') for i in range(5)]
        self.fish = [load_sample(f'histogram/fish_{i}') for i in range(5)]
        self.apple_test = load_sample(f'histogram/apple_test')
        self.fish_test = load_sample(f'histogram/fish_test')
        self.apple_incorrect = load_sample(f'histogram/apple_incorrect')
        self.fish_incorrect = load_sample(f'histogram/fish_incorrect')

histogram = Histogram()