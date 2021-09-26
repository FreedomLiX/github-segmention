from copy import copy
import cv2
from PIL import Image
import numpy as np
import random
import torch
from matplotlib import pyplot as plt


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[0:block_size, 0:block_size].clone()
        jigsaws[0:block_size, 0:block_size] = jigsaws[x * block_size:(x + 1) * block_size,
                                              y * block_size:(y + 1) * block_size].clone()
        jigsaws[x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


if __name__ == "__main__":
    image = Image.open("/home/dixn/fsdownload/resultsSegmentation/750.jpg")
    image = image.resize((448, 448))
    img = np.asarray(image)
    img = torch.from_numpy(img)
    # image = Image.open("/home/dixn/fsdownload/val/745.jpg")
    for n in [1, 2, 4, 8]:
        out = jigsaw_generator(img, n)
        plt.imshow(out)
        plt.savefig("750__" + str(n) + ".jpg")
        plt.show()
