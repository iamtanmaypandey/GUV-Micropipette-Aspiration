import cv2
import numpy as np

file = './frame1655.jpg'
img = cv2.imread(file,0)
imgo = cv2.imread(file)
imgoo = cv2.imread(file)
mean = np.mean(img)
for row in range(0,len(imgo)):
    for col in range(0,len(imgo[row])):
        for pix in range(0,len(imgo[row][col])):
            if int(imgo[row][col][pix] - mean) < 0:
                imgo[row][col][pix] = 0
            else:
                imgo[row][col][pix] = imgo[row][col][pix] - mean

og = cv2.resize(imgoo, (512,512),
               interpolation = cv2.INTER_LINEAR)
fil = cv2.resize(imgo, (512,512),
               interpolation = cv2.INTER_LINEAR)
                
#cv2.imshow('Original',og)
cv2.imwrite('Background Subtracted.tif',fil)
