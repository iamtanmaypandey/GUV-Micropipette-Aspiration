import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line
from scipy.signal import find_peaks
import cv2
from scipy.ndimage import gaussian_gradient_magnitude
from sklearn.metrics import mean_absolute_error as mbe
from matplotlib.widgets import PolygonSelector
import pandas as pd
import easygui as eg
from alive_progress import alive_bar

netrows = []
columns = ['xt','del xt','xp','del xp','xv','del xv','R','del R','S','del S','V','del V']

def getxp(file):
    fig,ax = plt.subplots()
    ax.imshow(file)
    rs = PolygonSelector(ax,eg.msgbox('Please Select the line which passes through the tip of pipette','SMBL-MPA'),useblit=False)
    plt.show()
    pt = rs.verts
    x,y = 0,0
    le = len(pt)
    for i in pt:
        x+= i[0]/le
        y+= i[1]/le
    return (x,y)

def getpoint(files,start,end,pix_size,theo_xp,rad,delr):
    files = files
    xp,y = getxp(files[0])
    xp = int(xp)
    with alive_bar(len(files)) as barr:
        for file in files:
            imk = cv2.cvtColor(file,cv2.COLOR_RGB2GRAY)
            ret,img = cv2.threshold(img,0,250,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
            ret,img = cv2.threshold(img,ret+50,250,cv2.THRESH_TOZERO)
            prof = gaussian_gradient_magnitude(profile_line(img,start,end,linewidth=1),sigma=3)
            peaks = find_peaks(prof)[0]
            prof2 = profile_line(imk,start,end,linewidth=1)
            peaks2 = find_peaks(prof2)[0]
            
            std = 10 #just a standard parameter for error calculation and mean
            
            #finding xt
            xt_test = peaks[0]
            print(xt_test)
            xt = 0
            xt_init = 0
            xt_pt = []
            xt_pd_pt = []
            for i in range(xt_test-std,xt_test+std):
                if i in peaks2:
                    xt_init += 1
                    xt += i
                    xt_pt.append(i)
            xt = xt/xt_init
            
            for j in range(0,len(xt_pt)):
                xt_pt[j] = xt_pt[j]
                xt_pd_pt.append(xt)
            xt_error = mbe(xt_pt,xt_pd_pt)    
            
            #finding xp
            xv_test = peaks[-1]
            xv = 0
            xv_init = 0
            xv_pt = []
            xv_id_pt = []
            for i in range(xv_test-std,xv_test+std):
                if i in peaks2:
                    xv_init += 1
                    xv += i
                    xv_pt.append(i)
            xv = xv/xv_init
            for j in range(0,len(xv_pt)):
                xv_pt[j] = xv_pt[j]
                xv_id_pt.append(xv)
            xv_error = mbe(xv_pt,xv_id_pt)
            
            xp_error = mbe([xp],[theo_xp]) 
            
            #finding desired parameters
            lv = abs(xp-xv)
            la = abs(xp-xt)
            del_lv = xp_error + xv_error
            del_la = xp_error + xt_error
            
            R = (lv**2 + rad**2)/(2*lv)
            delR = np.sqrt( ((rad**2)/(lv**2))*(delr**2) + (1/4)*(1- ((rad**2)/(lv**2)))*(del_lv**2))
            
            if la >= rad:
                s_in = np.pi*2*rad*la
                v_in = np.pi*(rad**2)*(la - (rad/3))
                
                del_s_in = 2*np.pi*np.sqrt((la**2)*(delr**2) + (rad**2)*(del_la**2))
                del_v_in = np.pi*rad*np.sqrt((rad**2)*(del_la**2) + (((2*la - rad))**2)*(delr**2))
                
            if (la >= 2*R - lv) and la<rad:
                s_in = np.pi*(la**2 + rad**2)
                v_in = (1/6)*np.pi*la*(3*(rad**2) + la**2)
                
                del_s_in = 2*np.pi*np.sqrt((la**2)*(del_la**2) + (rad**2)*(delr**2))
                del_v_in = 0.5*np.pi*((4*(rad**2)*(la**2)*(delr**2)) + (((rad**2 + la**2)**2)*(del_la**2)))
                
            else:
                s_in = 0
                v_in = 0
                del_s_in = 0
                del_v_in = 0
            
            s_out = np.pi*(lv**2 + rad**2)
            del_s_out = 2*np.pi*np.sqrt((lv**2)*(del_lv**2) + (rad**2)*(delr**2))
            
            v_out = (1/6)*np.pi*(lv)*(3*(rad**2) + lv**2)
            del_v_out = 0.5*np.pi*np.sqrt((4*(rad**2)*(lv**2)*(delr**2)) + ((rad**2 + lv**2)**2)*(del_lv**2))
            
            S = s_in + s_out
            V = v_in + v_out
            
            delS = del_s_in + del_s_out
            delV = del_v_in + del_v_out
            
            
            #entering all data into a csv file
            r = [xv,xv_error,xp,xp_error,xt,xt_error,R,delR,S,delS,V,delV]
            for inde in range(0,len(r)):
                r[inde] = float(r[inde])*float(pix_size)
            
            netrows.append(r)
            barr()
    df = pd.DataFrame(netrows,columns=columns)
    return df