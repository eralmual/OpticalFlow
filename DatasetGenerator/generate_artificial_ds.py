import cv2
import os


bgnd_dir = "/media/HDD/PARMA/Optical_flow/datasets/original_video/"  # Base dataset folder
ball_dir = "/media/HDD/PARMA/Optical_flow/datasets/sen_blured/"

save_dir = "/media/HDD/PARMA/Optical_flow/datasets/artificial/"      # Save folder


# Function to merge two datasets, recives the dir #1 and #2, asumes the second is 
# smaller or equal to the other so it'll be used ciclicaly
def mergeDatasets(dir1, dir2):

    # Load dataset file names and get its length
    ds1 = os.listdir(dir1)
    ds1_lenght = len(os.listdir(dir1))
    ds2 = os.listdir(dir2)
    ds2_lenght = len(os.listdir(dir2))



    for i in range(0, ds1_lenght):

        # Save the image names of the first dataset since it will be used many times
        img_name = ds1[i]

        # Load the images
        ds1_image = cv2.imread(dir1 + img_name)
        ds2_image = cv2.imread(dir2 + ds2[i % ds2_lenght])

        # If ds2_image has diferent dimensions to ds1_image then ds2 is resized
        if(ds1_image.shape != ds2_image.shape):
            ds2_image = cv2.resize(ds2_image, ds1_image.shape)
        
        # Overlay the two images together
        #new_image =  ds1_image + ds2_image - np.multiply(ds2_image, ds1_image)
        new_image = cv2.addWeighted(ds2_image, 1, ds1_image, 0.999, 0)

        # Save the image
        cv2.imwrite(save_dir + (ds1_lenght - len(str(i))) * '0' + str(i) + '.png', new_image)
