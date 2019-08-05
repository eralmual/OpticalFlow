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
x_dim_output = 512  # Width of the image
y_dim_output = 256  # Height of the image
circle_radius = 5   # Radius of the circle
cycles = 20         # Cycles per animation, gets modified when multiple animations are needed
original_dir = 'drive/My Drive/PARMA/OpticalFlow/GitRepo/OriginalVideo/'  #Dir of the original stt of images, modified for multiple animations
# Color coefficients, used for multiple animations
red = 1
green = 0
blue = 0
#Used to separate colors in a big animation
acc_red = 0
acc_green = 0
acc_blue = 0


def single_animation(frames, noise, trajectory, save_dir):

    ############################## ANIMATION #################################
    # Create the animation
    circle_animation = CircleAnimation(y_dim_output, 
                                x_dim_output, 
                                circle_radius,
                                cycles,
                                frames,
                                trajectory,
                                noise)

    # Gets the animation, add noise to it and limit
    final_animation = bgnd_animation(frames, circle_animation)
    final_animation = final_animation + np.round(np.random.normal(0, 10, final_animation.shape)).astype(np.int64)
    final_animation[final_animation < 0] = 0
    final_animation[final_animation > 255] = 255

    # Save final animation
    save_imgs(final_animation, frames, save_dir)
    # Save ground truth
    save_imgs(circle_animation.animation, frames, save_dir + 'gt/')
    # Save positions
    save_positions(circle_animation.positions[0], circle_animation.positions[1], save_dir, trajectory)

def save_positions(posx, posy, save_dir, trajectory, name=''):
    # Save csv positions
    df = pd.DataFrame(np.array([posx, posy]).transpose())
    df.to_csv(save_dir + 'Positions_' + trajectory + '.csv')

def save_imgs(images, frames, save_dir, name=''):

    # Count the digits of the amount of samples to cal the padding 0's
    samples_digits = len(str(frames))

    # Add a progress bar
    pbar = tqdm(desc='Saving images into: ' + save_dir, total=frames)

    for i in range(0, frames):
        # Save the image
        cv2.imwrite(save_dir + (samples_digits - len(str(i))) * '0' + str(i) + '.png', images[:, :, i])
        # Update the progress bar
        pbar.update(1)

    pbar.close()


# Recives the amount of generated frames and a CircleAnimation object
# Returns the combination of the bgnd and the animation
def bgnd_animation(frames, animation):

    ############################## LOAD BGND IMAGES #################################
    pbar = tqdm(desc='Loading background images...', total=frames)
    # Container to all bgnd images
    bgnd = np.zeros((y_dim_output, x_dim_output, frames, 3)) # The 3 is for rgb
    # Load all bgnd images 
    for i in range(0, frames):
        # Load the background image and resize it
        bgnd[:, :, i] = cv2.resize(cv2.imread(original_dir + str(i % 241) + '.png'), (x_dim_output, y_dim_output))
        # Update the progress bar
        pbar.update(1)

    pbar.close()


    ############################## COMBINING BGND AND ANIMATION #################################
    print("Combining background and animation...")
    # Container to all bgnd images
    images = np.zeros((y_dim_output, x_dim_output, frames, 3))
    # Calc each chanel of the final image
    images[:, :, :, 2] = (animation.blured_animation * red).astype(int) + bgnd[:, :, :, 2] - acc_red        # Red
    images[:, :, :, 1] = (animation.blured_animation * green).astype(int) + bgnd[:, :, :, 1] - acc_green    # Green
    images[:, :, :, 0] = (animation.blured_animation * blue).astype(int) + bgnd[:, :, :, 0] -acc_blue       # Blue
    # Limit the images 
    # Red
    images[images < 0] = 0
    images[images > 255] = 255
    
    # Return animation 
    return images


if __name__ == "__main__":

    # Parameters
    max_samples = 5000  # Sets the maximum amount of frames
    x_dim_output = 512  # Width of the image
    y_dim_output = 256  # Height of the image
    circle_radius = 5   # Radius of the circle
    cycles = 20         # Cycles per animation, gets modified when multiple animations are needed
    original_dir = '/hdd/Datasets/OpticalFlow/original_video/'  #Dir of the original stt of images, modified for multiple animations
    # Color coefficients, used for multiple animations
    red = 1
    green = 0
    blue = 0
    #Used to separate colors in a big animation
    acc_red = 0
    acc_green = 0
    acc_blue = 0

    step = 0
    std = 0
    trajectory = ''
    save_dir = ''

    try:
        # We want to recognize s,n,t,c as options with argument thats why 
        # the : follows them, h doesnt need arguments so its alone
        opts, args = getopt.getopt(sys.argv[1:],"hs:e:t:d:",["step=","eev=","trajectory=","dir="])
    except getopt.GetoptError:
        print('animator.py -s <step> -e <standar deviation> -t <trajectory> -d <save dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('animator.py -s <step> -e <standar deviation> -t <trajectory> -d <save dir>')
            sys.exit()

        elif opt in ("-s", "--step"):
            step = int(arg)

        elif opt in ("-e", "--eev"):
            std = int(arg)

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
    
   
   # TODO: EXPECT MULTIPLE TAJECTORIES SEPARATED BY +
    single_animation(  max_samples // step,
                        std,
                        trajectory,
                        save_dir)


