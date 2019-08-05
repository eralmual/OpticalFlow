import sys
import cv2
import copy
from cv2 import ximgproc
import matplotlib.pyplot as plt

from minimum_selection import minimum_selection
from nms import non_max_suppression

 
def selective_search(image_path, type):
 
    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)
 
    # read image
    im = cv2.imread(image_path)
    
    # resize image
    #newHeight = 200
    #newWidth = int(im.shape[1]*200/im.shape[0])
    #im = cv2.resize(im, (newWidth, newHeight))    
 
    # create Selective Search Segmentation Object using default parameters
    ss = ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
 
    # Switch to fast but low recall Selective Search method
    if (type == 'f'):
        ss.switchToSelectiveSearchFast()
 
    # Switch to high recall but slow Selective Search method
    elif (type == 'q'):
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    else:
        print(__doc__)
        sys.exit(1)
 
    # run selective search segmentation on input image
    rects = ss.process()
    # Select the smaller boxes
    #rects = minimum_selection(rects, 10)
    # Transform  data from [x y w h] -> [x1 y1 x2 y2]
    rects[:, 2:] += rects[:, 0:2]
    #rects = non_max_suppression(rects, 0.9)
    print('Total Number of Region Proposals: {}'.format(len(rects)))
     
    # number of region proposals to show
    numShowRects = 80
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 2

    print("#################### SELECTED ####################")
    print(rects)


    #out_folder = '/home/gerardo/Documents/Parma/Tracking/Object Detector/Results/40/'

    # create a copy of original image
    imOut = im.copy()

    # itereate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x1, y1, x2, y2 = rect
            cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            crop_img = im[y1:y2, x1:x2]
            #cv2.imwrite(out_folder + str(i) + ".png", crop_img)
        else:
            break

    # show output
    plt.imshow(imOut)
    plt.grid(True)
    plt.show()

    """
    # m is pressed
    if k == 109:
        # increase total number of rectangles to show by increment
        numShowRects += increment
    # l is pressed
    elif k == 108 and numShowRects > increment:
        # decrease total number of rectangles to show by increment
        numShowRects -= increment
    # q is pressed
    elif k == 113:
        break
    """
    

if __name__ == '__main__':
    selective_search("/hdd/Datasets/OpticalFlow/s10_e3_tsen/gt/025.png", 'f')