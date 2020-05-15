import numpy as np
import cv2
import czifile
import pickle
import matplotlib.pyplot as plt
import scipy.misc
import math
import random

DDEPTH = 5
NUM_LANDMARKS = 50


cell = czifile.imread('CAAX_100X_20171024_1-Scene-08-P8-B02.czi')


with open('cells.pickle', 'wb+') as f:
   pickle.dump(cell[0][0][3], f)


with open('cells.pickle', 'rb+') as f:
    cell_images = pickle.load(f)


for i in range(25, 50):
    image = cell_images[i]
    image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

    seed = (413, 806)

    cv2.floodFill(image, None, seedPoint=seed, newVal=(0, 0, 255), loDiff=(3, 3, 3, 3), upDiff=(5, 5, 5, 5))

    cv2.circle(image, seed, 2, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel, iterations=3)

    cv2.imshow(f'flood-{i}', image)
    cv2.imwrite(f'data/p08-1/p08-{i}.jpg', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Process finished")




