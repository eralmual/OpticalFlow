import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from scipy.signal import convolve2d
from motionFilter import motion_filter

class CircleAnimation():

    # Recives the dimensions of the matrix
    def __init__(self, height, width, radius, cycles, frames, trajectory, noise_std):

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
        self.animation = np.zeros((self.height, self.width, self.frames))
        self.blured_animation = np.zeros((self.height, self.width, self.frames))  

        # Start animation
        self.animate()

    def animate(self):
        filter_size = 11

        # Get trajectories
        posx = self.positions[0]
        posy = self.positions[1]

        # Add a progress bar
        pbar = tqdm(desc='Generating ' + self.trajectory + ' animation...', total=self.frames)

        for i in range(0, self.frames):
            x = posx[i]
            y = posy[i]
            # Animate position
            self.animation[:, :, i] = self.generate_circle(x, y)

            # Get the movement angle
            angle = np.arctan(y / x) * 180/np.pi
            f = motion_filter(filter_size, angle)
            self.blured_animation[:, :, i] = convolve2d(self.animation[:, :, i], 
                                                        f,  
                                                        boundary='symm')[0:self.height, 0: self.width]

            # Update the progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

    def generate_circle(self, x ,y):

        # Clears previous drawing
        matrix = np.zeros((self.offset_height, self.offset_width))
        # Draws a circle
        self.draw_circle(y, x, self.radius, matrix)
        # Fills the drawn circle
        self.fill_circle(y, x, matrix)
        # Crop the image to the needed dimensions
        return matrix[0:self.height, 0:self.width]


    def draw_circle(self, xc, yc, radius, matrix):
        d = 3 - (2 * radius)
        x = 0
        y = radius
        matrix[xc+x, yc+y] = 255
        matrix[xc-x, yc+y] = 255
        matrix[xc+x, yc-y] = 255
        matrix[xc-x, yc-y] = 255
        matrix[xc+y, yc+x] = 255
        matrix[xc-y, yc+x] = 255
        matrix[xc+y, yc-x] = 255
        matrix[xc-y, yc-x] = 255
        while x <= y:
            x+=1
            if d<0:
                d = d + (4*x) + 6
            else:
                d = d + 4 * (x - y) + 10
                y-=1
            matrix[xc+x, yc+y] = 255
            matrix[xc-x, yc+y] = 255
            matrix[xc+x, yc-y] = 255
            matrix[xc-x, yc-y] = 255
            matrix[xc+y, yc+x] = 255
            matrix[xc-y, yc+x] = 255
            matrix[xc+y, yc-x] = 255
            matrix[xc-y, yc-x] = 255

    def fill_circle(self, xc, yc, matrix):
        # Fills the circle
        if self.radius > 1:

            for i in range(1, self.radius): 
                self.draw_circle(xc, yc, i, matrix)

        matrix[xc, yc] = 255

    
    def generate_trajectory(self):

        # Define base movement in x, start, finish and amount of values
        x = np.linspace(1, self.width, self.frames)
        # Create gaussian noise with u=0 and std=noise and add it to x
        x += np.random.normal(0, self.noise_std, x.shape)
        
        # Define how many cicles will be done 1 cycle = 2pi
        cita = np.linspace(0, self.cycles * np.pi, self.frames)
        # Generate noise for the movement in y and center the values
        y = np.random.normal(0, self.noise_std, x.shape) + self.height // 2
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

        #THIS IS TO LIMIT MOVEMENT INSIDE THE TUBE
        # Set theposition limits
        y = np.clip(y, 95, 195)               # Empiric limits
        x = np.clip(x, 1, self.offset_width)  # Image limits
            
        return [x.astype(int), y.astype(int)]
