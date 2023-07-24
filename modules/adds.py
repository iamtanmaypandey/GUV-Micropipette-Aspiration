#importing required modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from matplotlib.widgets  import PolygonSelector
import easygui as eg
from skimage.measure import profile_line
import scipy
from alive_progress import alive_bar

apptitle = 'SMBL-MPA'

#defining helper functions
def threshs(pts,file):
    thresh = [(pts[0][1]+pts[1][1])/2,(pts[-1][1]+pts[-2][1])/2]
    pipette = [pts[0],pts[1]]
    x1 = pipette[0][0]
    x2 = pipette[1][0]
    y1 = 0
    y2 = np.shape(file)[0]
    return thresh,[(x1,y1),(x2,y2)]

def grayim(file):
    return cv2.cvtColor(file,cv2.COLOR_BGR2GRAY)

def calculatedist(pts):
    pts = [pts[0],pts[-1]] #contains data in form [y1,y2]
    return np.sqrt((pts[0]-pts[1])**2)

def calculateidealdist(pts):
    points = len(pts)
    top = pts[0:points//2]
    bottom = pts[points//2:points][::-1]
    dist = []
    average = 0
    for i in range(0,len(top)):
            topx = top[i][0]
            topy = top[i][1]
            bottomx = bottom[i][0]
            bottomy = bottom[i][1]
            dist.append(np.sqrt((topx-bottomx)**2+(topy-bottomy)**2))
    for j in dist:
        average += + j/len(dist)
    k = ((top[0][1] + bottom[0][1]) - (top[-1][1] + bottom[-1][1]))/((top[0][0] + bottom[0][0]) - (top[-1][0] + bottom[-1][0]))
    if k < 1:
        epsilon=(np.sqrt(1+k**2))
    else:
        epsilon=(np.sqrt(1+ 1/(k**2)))
    return average,k,epsilon

def getfixcoords(file):
    eg.msgbox(
        'Please select the 4 points as instructed: \n 1st and 4th : Start of the needle. \n 2nd & 3rd : Tip of the needle. \n Please form a rectangular box.',apptitle
    )
    fig,ax = plt.subplots()
    ax.imshow(file,cmap='gray')
    selector = PolygonSelector(ax, print('Yes'), useblit=True)
    plt.show()
    return selector.verts

#define main function

def syringeauto(files):
    files = files
    coords = getfixcoords(files[0])  #have to return the coords for system axis calculations and error also.
    os.system('cls')
    idealdiameter, k,epsilon = calculateidealdist(coords)
    threshy, pts = threshs(coords,files[0])
    x1,x2,y1,y2 = pts[0][0],pts[1][0],pts[0][1],pts[1][1]
    totaldistance = []
    netays = []
    netdiameter = 0
    netaxis = [0,0]
    print('Calculating \n Untill then please have a \u26FE .')
    with alive_bar(len(files)) as bar:
        for file in files:
            file2 = cv2.cvtColor(file,cv2.COLOR_RGB2GRAY)
            crop = file2[
                int(y1):int(y2),
                int(x1):int(x2)
            ]
            crop = cv2.bitwise_not(crop)
            dist = []
            avgy = []
            for j in range(0,len(crop[0])):
                start = (0,j)
                end = (y2,j)
                prof = profile_line(crop,start,end,linewidth=1)
                prof[0:int(threshy[0])] = 0
                prof[int(threshy[1]):] = 0
                #print(prof)
                peaks = scipy.signal.find_peaks(prof)
                #print(peaks)
                ys = [peaks[0][-1],peaks[0][0]]
                dist.append(calculatedist(peaks[0]))
                avgy.append(ys)
            avgdist = 0
            avgys = [0,0]
            for distance in dist:
                avgdist += distance/len(dist)
            totaldistance.append(avgdist)
            length_avgys = len(avgy)
            for yk in avgy:
                for p in range(0,2):
                    avgys[p] += yk[p]/length_avgys
            netays.append(avgys)
            bar()
    for distance in totaldistance:
        netdiameter += distance/len(totaldistance)
    for y in netays:
        for p in range(0,2):
            netaxis[p] += y[p]/len(netays)
    error = np.sqrt((netdiameter - idealdiameter)**2/idealdiameter**2)
    print('\u2705 Done !')
    return netdiameter,error,k,epsilon,netaxis,coords