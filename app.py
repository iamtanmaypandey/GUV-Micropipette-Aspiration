#loading the operating system library
import os
import art as art

os.system('cls')
art.tprint('MPA')
print('The Libraries are loading ..')

#importing the requried files
import numpy as np
import matplotlib.pyplot as plt
import easygui as eg
import cv2
from matplotlib.widgets import Button,Slider
import scipy.ndimage as nd
from alive_progress import alive_bar
from skimage.measure import profile_line
import pylab

#importing custom built modules
from modules.rotateimg import rotateimg
from modules.crop import crop
from modules.rollingball import rolling_ball_filter
from modules.removebg import removebg
from modules.adds import syringeauto
from modules.getpoints import getpoint

print('All libraries loaded :) ')

#setting app title
apptitle = 'MicroPipette Aspiraton'

#asking the system things
user = eg.multenterbox('Enter the parameters ', apptitle, ['pixel length in detector (\u03BC) :', 'Binning Size','Objective','Magnifier'])
pix = float(user[0])
binning = float(user[1])
objective = float(user[2])
magnifier = float(user[3])

#pixlen defining :

pixlen = pix*binning*magnifier/objective

#asking the file location
path = eg.diropenbox('Select the directory to process', apptitle)

#read the files in the directory and save it to a list
files = []
names = []
bg = []

print('Reading the files: ')
with alive_bar(len(os.listdir(path))) as bar:
    for file in os.listdir(path):
        p = (f'{path}/{file}')
        files.append(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB))
        names.append(file)
        bar()
print('\u2705 Read successfully')

ifbgremo = eg.ynbox('Should we do background removal by mean method?',apptitle, ('Yes','No'))
if ifbgremo:
    print('Background Subtraction Started')
    #read all files and update it in files list
    with alive_bar(len(files)) as bgsub:
        for ind in range(0,len(files)):
            rgb = files[ind]
            gray = cv2.cvtColor(files[ind],cv2.COLOR_RGB2GRAY)
            final = removebg(rgb,gray)
            files[ind] = final
            bgsub()
    print('\u2705 Background Subtraction finished')
 

#get rotation degree for each image    
deg_rotation = rotateimg(files[0])

#apply rotation to all images
print('Applying rotation to all images :)')
with alive_bar(len(files)) as bar2:
    for ind in range(0,len(files)):
        img = nd.rotate(files[ind],deg_rotation)
        files[ind] = img
        bar2()
print('\u2705 Rotation Applied. Successfully :)')    

#Cropping window
print('Please select the Region of Interest in next window')
roi = crop(files[0])

print('Applying ROI Selection to all Images')
with alive_bar(len(files)) as bar3:
    for ind in range(0,len(files)):
        files[ind] = files[ind][
            int(roi[1]):int(roi[1]+roi[3]),
            int(roi[0]):int(roi[0]+roi[2])
        ]
        bar3()
print('\u2705 Images Cropped Successfully :)')

isavecrop = eg.ynbox('Save the cropped images?', apptitle, ('Yes','No'),cancel_choice='No')
if isavecrop:
    path2save = eg.diropenbox('Select the directory to save the images',apptitle)
    with alive_bar(len(files)) as bar4:
        for i in range(0,len(files)):
            t = files[i]
            t = cv2.cvtColor(files[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{path2save}/rbg-{names[i]}',t)
            bar4()

#get number of rows and columns from roi
rows = np.shape(files[0])[0]
columns = np.shape(files[0])[1]

print('Calculating r and \u0394 r')
diameter_syringe,error,k,epsilon,axis,coords = syringeauto(files)
r = diameter_syringe/2

systemaxis = (coords[0][1] + coords[1][1] + coords[2][1] + coords[3][1])/4

theo_xp = (coords[1][0]+coords[2][0])/2

start_axis = (systemaxis,0) #(row,column)
end_axis = (systemaxis,columns)

print('Starting the calculations. \U0001F686 ')

#calculate datapoints
data_cal = getpoint(files,start_axis,end_axis,pixlen,theo_xp,r,error)

print('Calculation Done. \u2611 \n \u2708')

#ask for save locations
path_data = eg.filesavebox('Select the location to save file',apptitle,filetypes='\*.csv')

data_cal.to_csv(path_data)

art.tprint('Completed.')
