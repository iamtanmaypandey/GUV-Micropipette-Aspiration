import cv2
import numpy as np

def removebg(imgo,gray):
    mean = np.mean(gray)
    for row in range(0,len(imgo)):
        for col in range(0,len(imgo[row])):
            for pix in range(0,len(imgo[row][col])):
                if int(imgo[row][col][pix] - mean) < 0:
                    imgo[row][col][pix] = 0
                else:
                    imgo[row][col][pix] = imgo[row][col][pix] - mean
    return imgo