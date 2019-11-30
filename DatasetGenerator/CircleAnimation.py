import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2, time
import os.path
from scipy.signal import convolve2d
from motionFilter import motion_filter

class CircleAnimation():

    # Recives the dimensions of the matrix
    def __init__(self, work_dir, height, width, radius, cycles, frames, trajectory, noise_std):

        # define extra space for fading effects later
        self.offset_height = height + 25 + radius
        self.offset_width = width + 50 + radius
        
        # Store circle parameters
        self.height = height
        self.width = width
        self.radius = radius

        # Store animation parameters
        self.cycles = 2*cycles
        self.frames = frames
        self.trajectory = trajectory
        self.noise_std = noise_std
        self.positions = self.generate_trajectory()

        self.work_dir = work_dir



    def animate(self, color):
        filter_size = 11

        # Get trajectories
        posx = self.positions[0]
        posy = self.positions[1]

        # Count the digits of the amount of samples to cal the padding 0's
        samples_digits = len(str(self.frames))

        # Add a progress bar
        pbar = tqdm(desc='Generating ' + self.trajectory + ' animation', total=self.frames)

        for i in range(0, self.frames):
            x = posx[i]
            y = posy[i]

            #Prev img
            prev = np.zeros((self.height, self.width, 3))
            frame = np.zeros((self.height, self.width, 3))
            
            # Animate position
            # Draws a circle in animation, with the given coordinates and radius,
            # gives the color in order (b,g,r) and the thickness of the border   
            cv2.circle(frame, (x, y), self.radius, color , -1)

            # Check if a previous image exists
            if(os.path.exists(self.work_dir + str(i) + '_acc.png')):
                prev = np.asarray(cv2.imread(self.work_dir + str(i) + '_acc.png') , np.float64)

            # Add previous images and clip the values
            frame = cv2.addWeighted(frame, 0.9, prev, 0.9, 0)

            # Save the animated image and the gt
            cv2.imwrite(self.work_dir + str(i) + '_acc.png', frame)
            gt = frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]
            gt[gt > 255] = 255
            cv2.imwrite(self.work_dir + str(i) + '_gt.png', gt)


            # Get the movement angle
            angle = np.arctan(y / x) * 180/np.pi
            f = motion_filter(filter_size, angle)

            # Convolve each dimension of the camera
            frame[:, :, 0] = convolve2d(frame[:, :, 0], f, boundary='symm')[0:self.height, 0: self.width]
            frame[:, :, 1] = convolve2d(frame[:, :, 1], f, boundary='symm')[0:self.height, 0: self.width]
            frame[:, :, 2] = convolve2d(frame[:, :, 2], f, boundary='symm')[0:self.height, 0: self.width]

            # Save the blured animation
            cv2.imwrite(self.work_dir + str(i) + '.png', frame)

            # Update the progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()



    def generate_trajectory(self):
        a = int(time.time()) % 50
        # Define base movement in x, start, finish and amount of values
        x = np.linspace(a, self.width, self.frames)
        # Create gaussian noise with u=0 and std=noise and add it to x
        x += np.random.uniform(-self.noise_std, self.noise_std, x.shape)
        
        # Define how many cicles will be done 1 cycle = 2pi
        cita = np.linspace(0, self.cycles * np.pi, self.frames)
        # Generate noise for the movement in y and center the values
        y = np.random.uniform(-self.noise_std, self.noise_std, x.shape) + self.height // 2 
        # Range of movement inside the tube
        tube_range = 50


        # sen trajectory
        if(self.trajectory == 'sen'):
            # Add the sen component to the trajectory
            y += tube_range * np.sin(cita) 
        
        # Rabo e' chancho
        elif(self.trajectory == 'df'):
            # Add sin in x and cos in y to make a circle
            x += tube_range * np.sin(cita) 
            y += tube_range * np.cos(cita) 

        else:
            print('Trajectory not recognized')
            x = np.zeros(self.frames)
            y = np.zeros(self.frames)

        # THIS IS TO LIMIT MOVEMENT INSIDE THE TUBE
        # Set theposition limits
        y = np.clip(y, 95, 195)               # Empiric limits
        x = np.clip(x, 1, self.offset_width)  # Image limits
            
        return [x.astype(int), y.astype(int)]