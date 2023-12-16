#importing required modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from matplotlib.widgets import PolygonSelector, Button, Slider
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_laplace, gaussian_filter
import easygui as eg
from skimage.measure import profile_line
from scipy.signal import find_peaks
from alive_progress import alive_bar
import art as art
import sklearn.metrics as metrics
import pandas as pd

apptitle = 'Micropipette Aspiration - SMBL'

class image:
    def __init__(self,image,coords,p_tip):
        self.file = image
        self.rows = image.shape[0]
        self.cols = image.shape[1]
        self.x = self.cols
        self.y = self.rows
        self.inverted = cv2.bitwise_not(self.file)
        self.pradius = self.find_pipette_radius(coords,p_tip)
        self.xv , self.xt, self.dxv, self.dxt = self.get_parameters(coords,p_tip)
    
    def thresh_otsu(self,img):
        gray = self.gray(img)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh
    
    @staticmethod    
    def gray(img):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        return gray
    
    def find_pipette_radius(self,coords,p_tip):
        x,y = self.x, self.y
        row = y
        col = x
        gray = self.gray(self.file)
        
        y_top = 0
        y_bottom = 0
        for i in range(0,len(coords)//2):
            y_top += coords[i][1]
            y_bottom += coords[len(coords)-i-1][1]
        row_start = int(y_top/(i+1))
        row_end = int(y_bottom/(i+1))
        col_max = int(p_tip)
        
        radius = 0
        
        for cols in range(len(gray[0])):
            if cols >= col_max :
                break
            
            profile = profile_line(gray,(0,cols),(y,cols))
            peaks, _ = find_peaks(profile, height=0)
            
            peak = []
            for i in range(0,len(peaks)):
                if peaks[i] > row_start and peaks[i] < row_end:
                    peak.append(peaks[i]) 
            
            rad = (abs(peak[0] - peak[-1]))/2
            radius += rad
            
        radius = radius/cols
        return radius
    
    def slider_on_img(self,img,msg):
        fig,ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(msg)
        ax.set_xlabel('Drag the slider to adjust the line')
        ax.set_ylabel('Press Enter to confirm')
        ax.set_xticks([])
        ax.set_yticks([])    
        axcolor = 'lightgoldenrodyellow'
        axline = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
        sline = Slider(axline, '', 0, np.shape(img)[1], valinit=0)
        def update(val):
            line.set_xdata(sline.val)
            fig.canvas.draw_idle()
        sline.on_changed(update)
        line = ax.axvline(sline.val, color='r')
        plt.show()
        return int(sline.val)
    
    
    def get_parameters(self,coords,p_tip):
        y_top = [coords[0][1],coords[1][1]]
        y_bottom = [coords[2][1],coords[3][1]]
        
        mid = 0
        for i in range(0,len(y_top)):
            mid += (y_top[i] + y_bottom[i])/2
        mid = mid/len(y_top)
        
        mid = int(mid)
        
        imaag = self.thresh_otsu(self.inverted)
        
        prof_left = gaussian_gradient_magnitude(profile_line(imaag,(mid,0),(mid,p_tip)),sigma=10) #(row,col) = (y,x)
        prof_right = gaussian_gradient_magnitude(profile_line(imaag,(mid,p_tip),(mid,self.cols)),sigma=10)
        
        peaks_right = find_peaks(prof_right)[0]
        peaks_left = find_peaks(prof_left)[0]
        
        try:
            xvi = peaks_right[-1] + p_tip
        
        except:
            xvi = self.slider_on_img(self.file,'Select the end of vesicle')
            
        try:
            xti = peaks_left[0]
        except:
            xti = self.slider_on_img(self.file,'Select the tip of vesicle')
        
        try: 
            xvii = peaks_right[-2] + p_tip
            xtii = peaks_left[1]
        except:
            xvii = xvi
            xtii = xti
        
        xti = [xtii,xti]
        xvi = [xvii,xvi]

        dxv = np.std(xvi)/np.mean(xvi)
        dxt = np.std(xti)/np.mean(xti)
        print(dxv,type(dxv))
        return xvi[-1],xti[-1], dxv, dxt  #xv , xt, dxv, dxt
        
def get_pippete_radius(img):
    #ask user to select region of pipette in the image using polygon selector
    fig,ax = plt.subplots()
    ax.imshow(img)
    ps = PolygonSelector(ax,eg.msgbox('Select the region of pipette',apptitle),useblit=False)
    plt.show()
    #get the coordinates of the selected region
    coords = ps.verts
    return coords

def get_pippete_tip(img):
    #plot a line on img and ask user to select the tip of pipette by dragging slider and adjusting the line
    eg.msgbox('Please select the line which passes through pipette tip',apptitle)
    fig,ax = plt.subplots()
    ax.imshow(img)
    ax.set_title('Select the tip of pipette')
    ax.set_xlabel('Drag the slider to adjust the line')
    ax.set_ylabel('Press Enter to confirm')
    ax.set_xticks([])
    ax.set_yticks([])
    axcolor = 'lightgoldenrodyellow'
    axline = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    sline = Slider(axline, '', 0, np.shape(img)[1], valinit=0)
    def update(val):
        line.set_xdata(sline.val)
        fig.canvas.draw_idle()
    sline.on_changed(update)
    line = ax.axvline(sline.val, color='r')
    plt.show()
    return int(sline.val)
    

def operate(files):  
    pippete_radius = []
    d_pippete_radius = 0
    
    xvs = []
    xts = []
    dxvs = []
    dxts = []
    
    la = []
    lv = []
    d_la = []
    d_lv = []
    
    coords = get_pippete_radius(files[0])
    p_tip = get_pippete_tip(files[0])
    os.system('cls')
    art.tprint('SMBL - MPA')
    print('\n','Initializing..')
    with alive_bar(len(files)) as bar2:
        for file in files:
            img = image(file,coords,p_tip)
            
            #finding pipette radius
            ra = img.pradius
            pippete_radius.append(ra)
            
            #finding the profile plots
            xv = img.xv
            xt = img.xt
            xvs.append(xv)
            xts.append(xt)
            dxvs.append(img.dxv)
            dxts.append(img.dxt)
            bar2()
    d_pippete_radius = np.std(pippete_radius)/np.mean(pippete_radius)
    print(d_pippete_radius,type(d_pippete_radius))
    print('Done','\n')
    
    print(f'Calculating Parameter 1')
    with alive_bar(len(xvs)) as bar3:
        for i in range(0,len(xvs)):
            lv.append(abs(p_tip - xvs[i]))
            d_lv.append(dxvs[i] + d_pippete_radius)
            bar3()
    print('Done','\n')
    print(f'Calculating Parameter 2')
    with alive_bar(len(xts)) as bar4:    
        for i in range(0,len(xts)):
            la.append(abs(p_tip - xts[i]))
            d_la.append(dxts[i] + d_pippete_radius)
            bar4()
        
    Radius_of_vesicle = []
    del_rov = []
    print('Done','\n')
    print(f'Calculating Parameter 3')
    with alive_bar(len(lv)) as bar5:
        for i in range(0,len(lv)):
            Lvs = lv[i]
            rs = pippete_radius[i]
            d_Lvs = d_lv[i]
            d_rs = d_pippete_radius
            
            rr = (Lvs**2 + rs**2)/(2*Lvs)
            Radius_of_vesicle.append(rr)
            
            d_rr = np.sqrt(((rs**2)/(Lvs**2)) * np.square(d_rs) + (Lvs**2 - rs**2)/(4*(Lvs**2)) * np.square(d_Lvs))
            del_rov.append(d_rr)
            bar5()
    print('Done','\n')
    print(f'Calculating Parameter 4')
    
    #calculating inner Surface area
    inner_surface_area = []
    del_isa = []
    
    with alive_bar(len(lv)) as bar6:
        for i in range(0,len(lv)):
            LA = la[i]
            LV = lv[i]
            d_LA = d_la[i]
            d_LV = d_lv[i]
            rr = pippete_radius[i]
            d_rr = d_pippete_radius
            
            if LA >= rr:
                inner_surface_area.append(2*np.pi*rr*LA)
                del_isa.append(2*np.pi*np.sqrt((LA**2)*np.square(d_rr) + (rr**2)*np.square(d_LA)))
            else:
                inner_surface_area.append(np.pi*(LA**2 + rr**2))
                del_isa.append(2*np.pi*np.sqrt((LA**2)*np.square(d_LA) + (rr**2)*np.square(d_rr)))
            bar6()

    print('Done','\n')
    print(f'Calculating Parameter 5')
    
    inner_volume = []
    d_iv = []
    
    with alive_bar(len(lv)) as bar7:
        for i in range(0,len(lv)):
            LA = la[i]
            LV = lv[i]
            d_LA = d_la[i]
            d_LV = d_lv[i]
            rr = pippete_radius[i]
            d_rr = d_pippete_radius
            
            if LA >= rr:
                inner_volume.append(np.pi*(rr**2)*(LA - (rr/3)))
                d_iv.append(np.pi*rr*(np.sqrt( (rr**2)*np.square(d_LA) + ((2*LA - rr)**2) * np.square(d_rr) ) ) )
            
            else:
                inner_volume.append((1/6)*(np.pi*LA)*(3*(rr**2) + LA**2))
                d_iv.append( (np.pi/2)*np.sqrt( 4*(rr**2)*(LA**2)*np.square(d_rr) + ((rr**2 + LA**2)**2)*np.square(d_LA) ) )
            
            bar7()
    
    print('Done','\n')
    print(f'Calculating Parameter 6')
        
    outer_surface_area = []
    d_osa = []
    with alive_bar(len(lv)) as bar8:
        for i in range(0,len(lv)):
            LA = la[i]
            LV = lv[i]
            d_LA = d_la[i]
            d_LV = d_lv[i]
            rr = pippete_radius[i]
            d_rr = d_pippete_radius
                
            ar = np.pi*(LV**2 + rr**2)
            outer_surface_area.append(ar)
                
            d_ar = 2*np.pi*np.sqrt((LV**2)*np.square(d_LV) + (rr**2)*np.square(d_rr))
            d_osa.append(d_ar)
            bar8()
        
    print('Done','\n')
    print(f'Calculating Parameter 7')
        
    outer_volume = []
    d_ov = []
    with alive_bar(len(lv)) as bar9:
        for i in range(0,len(lv)):
            LA = la[i]
            LV = lv[i]
            d_LA = d_la[i]
            d_LV = d_lv[i]
            rr = pippete_radius[i]
            d_rr = d_pippete_radius
                
            vol = (1/6)*(np.pi*LV)*(3*(rr**2) + LV**2)
            outer_volume.append(vol)
                
            d_vol = (1/2)*np.pi*np.sqrt(4*(rr**2)*(LV**2)*np.square(d_rr) + ((rr**2 + LV**2)**2)*np.square(d_LV))
            d_ov.append(d_vol)
            bar9()
        
    print('Done','\n')
        
    print('All Parameters Calculated Successfully')
        
    os.system('cls')
    art.tprint('SMBL')
        
    print('\n','Creating CSV File')
    colum = ['Pipette Radius','Delta Pipette Radius','Vesicle End','Vesicle Tip','Delta Vesicle End','Delta Vesicle Tip','Radius of Vesicle','Delta Radius of Vesicle','Inner Surface Area','Delta Inner Surface Area','Inner Volume','Delta Inner Volume','Outer Surface Area','Delta Outer Surface Area','Outer Volume','Delta Outer Volume']
    df = pd.DataFrame(np.empty((0,len(colum))))
    df.columns = colum
    print('/n','Writing the data in a file')
    with alive_bar(len(pippete_radius)) as bar10:    
        for i in range(0,len(pippete_radius)):
            df.loc[i] = [pippete_radius[i],d_pippete_radius,xvs[i],xts[i],dxvs[i],dxts[i],Radius_of_vesicle[i],del_rov[i],inner_surface_area[i],del_isa[i],inner_volume[i],d_iv[i],outer_surface_area[i],d_osa[i],outer_volume[i],d_ov[i]]
            bar10()
    
    print('Done','\n')
    print('Saving the file')
    
    location = eg.diropenbox('Select the directory to save the CSV file', apptitle)
    df.to_csv(f'{location}/data.csv')

#files = []
#lo = eg.diropenbox('Select the directory to process', apptitle)
#for file in os.listdir(lo):
#    p = (f'{lo}/{file}')
#    files.append(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB))
#files = files[0:1]
#operate(files)