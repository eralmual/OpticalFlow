import math
import numpy as np
from skimage.transform import rotate

def motion_filter(length = 9, angle = 0):

    if(length % 2 == 1):
        size = [length, length]
    else:
        size = [length + 1, length + 1]

    # First generate a horizontal line across the middle
    f = np.zeros(size)
    f[math.floor(length / 2), 0:length] = 1

    # Then rotate to specified angle
    # Specify bilinear interpolation and that we dont want to keep the image clipped
    f = rotate(f, angle, order=1)
    f = f / np.sum(f)

    return f  