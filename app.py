#importing the requried files
import numpy as np
import matplotlib.pyplot as plt
import easygui as eg
import cv2
import os
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
from modules.syringe import syringe
#from modules.guv import guv

#setting app title
apptitle = 'MicroPipette-SMBL'

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
print('Read successfully')

#Show user the line profile at the center of image
initfile = cv2.cvtColor(files[0],cv2.COLOR_BGR2GRAY)
shape = initfile.shape
#print(shape)
start = (0,shape[0]//2)
end = (shape[1], shape[0]//2)
profile = profile_line(initfile, start,end,linewidth=1)

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
    print('Background Subtraction finished')

#get rotation degree for each image    
deg_rotation = rotateimg(files[0])

#apply rotation to all images
print('Applying rotation to all images :)')
with alive_bar(len(files)) as bar2:
    for ind in range(0,len(files)):
        img = nd.rotate(files[ind],deg_rotation)
        files[ind] = img
        bar2()
print('Rotation Applied. Successfully :)')    

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
print('Images Cropped Successfully :)')

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

pixlen = eg.enterbox('Enter pixel length:', apptitle)
pixwid = eg.enterbox('Enter pixel width:', apptitle)

diameter_syringe,epsilon = syringe(files)
print(diameter_syringe)