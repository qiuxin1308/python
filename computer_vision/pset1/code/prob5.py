## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2


# Fill this out
def kernpad(K,size):
    Ko = np.zeros(size,dtype=np.float32)
    # Placeholder code
    k_y = K.shape[0]
    k_x = K.shape[1]
    hk_y = k_y//2+1
    hk_x = k_x//2+1
    k_1 = np.zeros((hk_y,hk_x),dtype=np.float32)
    k_2 = np.zeros((hk_y,hk_x-1),dtype=np.float32)
    k_3 = np.zeros((hk_y-1,hk_x),dtype=np.float32)
    k_4 = np.zeros((hk_y-1,hk_x-1),dtype=np.float32)
    k_1,k_2,k_3,k_4 = K[0:hk_y,0:hk_x],K[0:hk_y,hk_x:k_x],K[hk_y:k_y,0:hk_x],K[hk_y:k_y,hk_x:k_x]
    k_1,k_2,k_3,k_4 = k_1[::-1,::-1],k_2[::-1,::-1],k_3[::-1,::-1],k_4[::-1,::-1]
    Ko[0:hk_y,0:hk_x],Ko[0:hk_y,size[1]-hk_x+1:size[1]],Ko[size[0]-hk_y+1:size[0],0:hk_x],Ko[size[0]-hk_y+1:size[0],size[1]-hk_x+1:size[1]] = k_1,k_2,k_3,k_4
    
    return Ko

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p5_inp.jpg')))/255.

# Create Gaussian Kernel
x = np.float32(range(-21,22))
x,y = np.meshgrid(x,x)
G = np.exp(-(x*x+y*y)/2/9.)
G = G / np.sum(G[:])


# Traditional convolve
v1 = conv2(img,G,'same','wrap')

# Convolution in Fourier domain
G = kernpad(G,img.shape)
v2f = np.fft.fft2(G)*np.fft.fft2(img)
v2 = np.real(np.fft.ifft2(v2f))

# Stack them together and save
out = np.concatenate([img,v1,v2],axis=1)
out = np.minimum(1.,np.maximum(0.,out))

imsave(fn('outputs/prob5.jpg'),out)


                 
