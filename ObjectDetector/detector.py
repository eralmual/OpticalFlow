import sys
import cv2
from cv2 import ximgproc
 
def selective_search(image_path, type):
 
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(8);
 
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
    print('Total Number of Region Proposals: {}'.format(len(rects)))
     
    # number of region proposals to show
    numShowRects = 80
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 2
 
    while True:

        out_folder = '/home/gerardo/Documents/Parma/Tracking/Object Detector/Results/40/'

        # create a copy of original image
        imOut = im.copy()
 
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                crop_img = im[y:(y+h), x:(x+w)]
                cv2.imwrite(out_folder + str(i) + ".png", crop_img)
            else:
                break
 
        # show output
        cv2.imshow("Output", imOut)
 
        # record key press
        k = cv2.waitKey(0) & 0xFF
 
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
            
    # close image show window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    selective_search("/home/gerardo/Documents/Parma/Tracking/Object Detector/Test_Images/40.png", 'q')