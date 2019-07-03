import cv2

print(cv2.__version__)
vidcap = cv2.VideoCapture('/media/HDD/PARMA/Optical_flow/Optical-Flow-for-VAD/videos/cut127.MTS')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("/media/HDD/PARMA/Optical_flow/datasets/original_bw/%d.png" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
