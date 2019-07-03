"""

def generate_double_animation(step, noise, trajectory, save_dir):

    # Get the selected trajectory positions
    max_samples = 5000
    circle_radius = 5
    samples = int(max_samples / step)
    posx, posy = get_trajectory(trajectory, max_samples, step, noise)

    # Get both trajectory
    t0x = posx[0]
    t0y = posy[0]
    t1x = posx[1]
    t1y = posy[1]

    # Count the digits of the amount of samples to cal the padding 0's
    samples_digits = len(str(max_samples))

    # Add a progress bar
    pbar = tqdm(desc='Generating images', total=samples)

    for i in range(0, samples):
        ###################### Artificial plot generation ######################
        # Create the circle with center in (x,y) and radius 100
        ball0 = generate_ball(t0x[i], t0y[i], circle_radius, max_samples)
        ball1 = generate_ball(t1x[i], t1y[i], circle_radius, max_samples)

        
        # Generate the fillter, convolve the image and invert them
        # Get the angle in degrees
        angle = np.arctan(t0x[i] / t0y[i]) * 180/np.pi
        f = motion(11, angle)
        ball0 = convolve2d(ball0, f,  boundary='symm')[0:256, 0: 512]

        angle = np.arctan(t1x[i] / t1y[i]) * 180/np.pi
        f = motion(11, angle)
        ball1 = convolve2d(ball1, f,  boundary='symm')[0:256, 0: 512]



        ###################### Add images together ######################
        # Load the background image
        bgnd = cv2.imread(original_dir + str(i % 241) + '.png')

        # If bgnd has diferent dimensions to ball then bgnd is resized
        if(ball0.shape != bgnd.shape):
            bgnd = cv2.resize(bgnd, (ball0.shape[1], ball0.shape[0]))


        # Overlay the two images together and add noise
        green = ball0 + bgnd[:,:,1]                           # Get the green component of the bgnd and add the ball
        green[green > 255] = 255                              # Limit the values to 255

        blue = ball1 + bgnd[:,:,2] - np.multiply(ball0, bgnd[:,:,2]) # Get the blue component of the bgnd and add the ball
        blue[blue > 255] = 255                                # Limit the values to 255
        img = np.array(bgnd, dtype=np.float64)                                  
        img[:,:,1] = green                                    # Add the balls
        img[:,:,2] = blue
        img = img + np.round(np.random.normal(0, 10, img.shape)).astype(np.int64)        # Add noise
        # Normalize
        img[img < 0] = 0
        img[img > 255] = 255

        # Save the plot
        cv2.imwrite(save_dir + 'gt/pos0' + (samples_digits - len(str(i))) * '0' + str(i) + '.png', ball0)
        cv2.imwrite(save_dir + 'gt/pos1' + (samples_digits - len(str(i))) * '0' + str(i) + '.png', ball1)
        cv2.imwrite(save_dir + (samples_digits - len(str(i))) * '0' + str(i) + '.png', img)
        #scipy.misc.imsave(save_dir + (samples_digits - len(str(i))) * '0' + str(i) + '.png', ball)

        # Close the figure to avoid memory leaks
        plt.close()

        # Update the progress bar
        pbar.update(1)

    pbar.close()

    # Save csv positions
    df = pd.DataFrame(np.array([t0x, t0y]).transpose())
    df.to_csv(save_dir + 'Positions_0_' + trajectory[:3] + '.csv')

    df = pd.DataFrame(np.array([t1x, t1y]).transpose())
    df.to_csv(save_dir + 'Positions_1_' + trajectory[3:] + '.csv')

"""