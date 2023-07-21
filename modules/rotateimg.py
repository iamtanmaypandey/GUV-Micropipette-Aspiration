import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,Slider
im = [0]

def retTru(a):
    return True

def rotate(deg):
    img = im[0]
    img = ndimage.rotate(img,deg)
    axs.imshow(img,cmap='gray')

fig, axs = plt.subplots()
fig.subplots_adjust(left=0.25, bottom=0.25)

def rotateimg(file):
    
    file = file
    im[0] = file
    axs.imshow(file,cmap='gray')
    axs.set_title('Close the window to confirm selection')
    axfreq = fig.add_axes([0.25, 0.1, 0.55, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label='Rotation Degrees',
        valmin=0,
        valmax=360
    )
    freq_slider.on_changed(rotate)
    plt.show()
    return freq_slider.val