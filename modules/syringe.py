import cv2
import numpy as np
import pylab as plt
from matplotlib.widgets import Button,Slider
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets  import PolygonSelector
from skimage.measure import profile_line

final = []
epsilon_net = []
sysaxs = [[0,0],[0,0],[0,0],[0,0]]

def line_select_callback(eclick):
    #x2, y2 = erelease.xdata, erelease.ydata
    return eclick

def syringe(files):
    filelen = len(files)
    for file in files: 
        fig, ax = plt.subplots()   
        ax.imshow(file)
        rs = PolygonSelector(ax, line_select_callback,useblit=False )        
        plt.show()
        points = len(rs.verts)
        top = rs.verts[0:points//2]
        bottom = rs.verts[points//2:points][::-1]
        average = 0
        dist = []
        sysax = [[top[0][0],top[0][1]],[bottom[0][0],bottom[0][1]],[top[-1][0],top[-1][1]],[bottom[-1][0],bottom[-1][1]]]
        for i in range(0,len(top)):
            topx = top[i][0]
            topy = top[i][1]
            bottomx = bottom[i][0]
            bottomy = bottom[i][1]
            dist.append(np.sqrt((topx-bottomx)**2+(topy-bottomy)**2))
        print(f'The coordinates found were: {top,bottom}')
        for j in dist:
            average = average + j/len(dist)
        final.append(average)
        k = ((top[0][1] + bottom[0][1]) - (top[-1][1] + bottom[-1][1]))/((top[0][0] + bottom[0][0]) - (top[-1][0] + bottom[-1][0]))
        if k < 1:
            epsilon_net.append(np.sqrt(1+k**2))
        else:
            epsilon_net.append(np.sqrt(1+ 1/(k**2)))
        #b = (top[0][1] + bottom[0][1])/2 - k*((top[0][0] + bottom[0][0])/2)
        pts = [[((top[0][0] + bottom[0][0])/2 ), ((top[0][1] + bottom[0][1])/2)],[((top[-1][0] + bottom[-1][0])/2) , ((top[-1][1] + bottom[-1][1])/2)]]
        fig,axs = plt.subplots(1,2)
        file2 = cv2.cvtColor(file,cv2.COLOR_RGB2GRAY)
        axs[0].imshow(file)
        endcol = np.shape(file)[1]
        x = [0,endcol]
        y = [pts[0][1],pts[1][1]]
        #axs[0].plot(x,y,'w')
        start = (pts[0][1],0)
        end = (pts[1][1],endcol)
        axs[1].plot(profile_line(file2,start,end,linewidth=1))
        plt.show()
    netav =  0
    epsilon = 0
    final_len = len(final)
    for dists in final:
        netav = netav + dists/final_len
    epsilon_len = len(epsilon_net)
    for eps in epsilon_net:
        epsilon = epsilon + eps/epsilon_len    
    return netav, epsilon
