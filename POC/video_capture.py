import cv2, os

# Capturing image from video
vidcap = cv2.VideoCapture('supermarket.mp4')
success,image = vidcap.read()
count = 0
images = []
success = True
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  images += ["frame"+count+".jpg"]
  count += 1

# image list
frame = cv2.imread(images[0])
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter('supermarket_out.mp4', fourcc, 30.0, (width, height))

for image in images:
    frame = cv2.imread(image)
    out.write(frame) # Write out frame to video
    cv2.imshow('video',frame)

cv2.destroyAllWindows()
out.release()
