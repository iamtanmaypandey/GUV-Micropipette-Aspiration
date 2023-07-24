import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from matplotlib.widgets  import PolygonSelector
import easygui as eg
from skimage.measure import profile_line
import scipy

apptitle = 's'
files = []
for file in os.listdir('./dataf/'):
    files.append(cv2.imread(f'./dataf/{file}'))


def line_select_callback(eclick):
    #x2, y2 = erelease.xdata, erelease.ydata
    return eclick

def syringe(files):
    setfile = files[-1]
    eg.msgbox(
        'Please select the 5 points as instructed: \n 1st and 5th : Start of the needle. \n 2nd & 4th : Tip of the needle  \n 3rd: End of Guv'
    )
    fig,ax = plt.subplots()
    ax.imshow(files[-1],cmap='gray')
    selector = PolygonSelector(ax, print('Yes'), useblit=True)
    plt.show()
    pts = selector.verts
    threshy = [(pts[0][1]+pts[1][1])/2,(pts[-1][1]+pts[-2][1])/2]
    pipette = [pts[0],pts[1]]
    x1 = pipette[0][0]
    x2 = pipette[1][0]
    y1 = 0
    file = cv2.cvtColor(files[-1],cv2.COLOR_BGR2GRAY)
    y2 = np.shape(file)[0]
    crop = file[
        int(y1):int(y2),
        int(x1):int(x2)
    ]
    crop = cv2.bitwise_not(crop)
    #crop = ndimage.gaussian_laplace(crop,sigma=1)
    cv2.imshow('laplace',crop)
    cv2.waitKey(0)
    for j in range(0,len(crop[0])):
        start = (0,j)
        end = (y2,j)
        prof = profile_line(crop,start,end,linewidth=1)
        
        prof[0:int(threshy[0])] = 0
        prof[int(threshy[1]):] = 0
        peaks = scipy.signal.find_peaks(prof)
    fig,ax = plt.subplots(1,4)
    ax[0].imshow(files[-1],cmap='gray')
    ax[1].plot([j,j],[0,y2],'b')
    ax[3].imshow(np.transpose(crop),cmap='gray')
    ax[1].imshow(crop, cmap='gray')
    ax[2].plot(prof)
    print(pts) 
    plt.show()
#print(files)

syringe(files)