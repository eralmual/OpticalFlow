import matplotlib.pyplot as plt
import numpy as np
import os

matrix = np.zeros((256, 512, 3))

def draw_circle(xc,yc,r):
    
    d = 3 - (2 * r)
    x = 0
    y = r

    matrix[xc+x, yc+y, :] = 255
    matrix[xc-x, yc+y, :] = 255
    matrix[xc+x, yc-y, :] = 255
    matrix[xc-x, yc-y, :] = 255
    matrix[xc+y, yc+x, :] = 255
    matrix[xc-y, yc+x, :] = 255
    matrix[xc+y, yc-x, :] = 255
    matrix[xc-y, yc-x, :] = 255
    while x <= y:
        x+=1
        if d<0:
            d = d + (4*x) + 6#oop /MAC R1, R2, R3
        else:
            d = d + 4 * (x - y) + 10
            y-=1
        matrix[xc+x, yc+y, :] = 255
        matrix[xc-x, yc+y, :] = 255
        matrix[xc+x, yc-y, :] = 255
        matrix[xc-x, yc-y, :] = 255
        matrix[xc+y, yc+x, :] = 255
        matrix[xc-y, yc+x, :] = 255
        matrix[xc+y, yc-x, :] = 255
        matrix[xc-y, yc-x, :] = 255

def fill_circle(xc,yc,r):
    # Fills the circle
    if r > 1:

        for i in range(1, r): 
            draw_circle(xc, yc, i)

    matrix[xc, yc, :] = 255

    plt.imshow(matrix)
    plt.grid(True)
    plt.show()
