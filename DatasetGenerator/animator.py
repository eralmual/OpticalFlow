#!/usr/bin/python3

import sys, getopt, math, os, cv2

import pandas as pd
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from tqdm import tqdm

from CircleAnimation import CircleAnimation

# Parameters
max_samples = 5000  # Sets the maximum amount of frames
x_dim_output = 256  # Width of the image
y_dim_output = 256  # Height of the image
circle_radius = 5   # Radius of the circle



#original_dir = '/content/drive/My Drive/PARMA/OpticalFlow/GitRepo/Dataset/OriginalVideo/'  #Dir of the original stt of images, modified for multiple animations
original_dir = '/home/erick/googleDrive/PARMA/OpticalFlow/GitRepo/Dataset/OriginalVideo/'




# Color coefficients, used for multiple animations
colors = ['b', 'g', 'r', 'c', 'm', 'y','w', 'k']
colors_val = [(255,0,0), (0, 255, 0), (0, 0, 255), (255,255,0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (0,0,0) ]

def single_animation(frames, noise, trajectory, save_dir, bgnd_dir):

    work_dir = save_dir + 'work/'

    ############################## ANIMATION #################################
    # Create the animation
    circle_animation = CircleAnimation( work_dir,
                                        y_dim_output, 
                                        x_dim_output, 
                                        circle_radius,
                                        int(abs(np.random.normal(0, 50, 1)[0])),
                                        frames,
                                        trajectory,
                                        noise)

    circle_animation.animate(colors_val.pop(0))
    # Combine the bgnd and the animation
    bgnd_animation(frames, bgnd_dir, work_dir, save_dir)
    # Save positions
    save_positions(circle_animation.positions[0], circle_animation.positions[1], save_dir, trajectory)



def save_positions(posx, posy, save_dir, trajectory, name=''):

    # Write the color of the ball in the name
    color = colors.pop(0)

    # Save csv positions
    df = pd.DataFrame(np.array([posx, posy]).transpose())
    df.to_csv(save_dir + 'Positions_' + trajectory + '_' + color +'.csv')

    
# Recives the amount of generated frames and a CircleAnimation object
# Returns the combination of the bgnd and the animation
def bgnd_animation(frames, bgnd_dir, work_dir, save_dir):

    ############################## LOAD BGND IMAGES and combine with animation #################################
    pbar = tqdm(desc='Mixing background and animation...', total=frames)
    # Container to all bgnd images and final images
    bgnd = np.zeros((y_dim_output, x_dim_output, 3))          
    img = np.zeros((y_dim_output, x_dim_output, 3))
    # Load all bgnd images 
    for i in range(0, frames):

        # Load the background image and resize it
        bgnd = cv2.resize(cv2.imread(bgnd_dir + str(i % 241) + '.png'), (x_dim_output, y_dim_output))
        # Load the animation img
        anim = cv2.resize(cv2.imread(work_dir + str(i % 241) + '.png'), (x_dim_output, y_dim_output))

       # Add previous images and clip the values
        img = cv2.addWeighted(bgnd, 0.9, anim, 0.9, 0)

        # Store the img
        cv2.imwrite(save_dir + str(i) + '.png', img)

        # Update the progress bar
        pbar.update(1)

    pbar.close()



if __name__ == "__main__":
    
    step = 0
    std = 0
    trajectory = ''
    save_dir = ''
    num_circles = 0

    try:
        # We want to recognize s,n,t,c as options with argument thats why 
        # the : follows them, h doesnt need arguments so its alone
        opts, args = getopt.getopt(sys.argv[1:],"hs:e:t:d:n:",["step=","eev=","trajectory=","dir=", "num="])
    except getopt.GetoptError:
        print('animator.py -s <step> -e <standar deviation> -t <trajectory> -d <save dir> -n <number of circles>\nThe given directory must have one inner directorie called work/')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('animator.py -s <step> -e <standar deviation> -t <trajectory> -d <save dir> -n <number of circles>\nThe given directory must have two inner directories called gt/ and work/')
            sys.exit()

        elif opt in ("-s", "--step"):
            step = int(arg)

        elif opt in ("-e", "--eev"):
            std = int(arg)

        elif opt in ("-n", "--num"):
            num_circles = int(arg)

        elif opt in ("-t", "--trajectory>"):
            if((arg == 'sen') or (arg == 'df') or (arg == 'sensen') or (arg == 'sendf')):
                trajectory = arg
            else:
                print('Unavailable trajectory, please choose sen or rc')
                sys.exit()
        
        elif opt in ("-d", "--dir"):

            save_dir = arg

            if(arg[-1] != '/'):
                save_dir += '/'


    frames = max_samples // step
    print("Using a step of " + str(step) + " will result in " + str(frames) + " frames")
    print("Using a standar deviation of: " + str(std))
    print("Drawing a " + trajectory + " trajectory")
    print("Using " + save_dir + " as save directory")

    
    # python3 animator.py -s 3 -e 2 -t sen -d /home/erick/googleDrive/PARMA/OpticalFlow/GitRepo/Dataset/test/ -n 5
   
    for i in range(0, num_circles):
        print('------------------------------------------------------- Adding circle #' + str(i) +' -------------------------------------------------------')
        single_animation(   max_samples // step,
                            std,
                            trajectory,
                            save_dir,
                            original_dir)
