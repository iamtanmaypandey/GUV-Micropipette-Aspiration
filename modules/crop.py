import cv2
import numpy

def crop(file):
    file = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
    crop = cv2.selectROI(file)
    return crop