## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
	Dx = [[1,0,-1],[2,0,-2],[1,0,-1]]
	Dy = np.transpose(Dx)
	#placeholder
	H = np.zeros(X.shape,dtype=np.float32)
	theta = np.zeros(X.shape,dtype=np.float32)
	Ix = conv2(X,Dx,mode='same',boundary='symm')
	Iy = conv2(X,Dy,mode='same',boundary='symm')
	#base on lec3 slide
	H = np.sqrt(np.square(Ix)+np.square(Iy))
	theta = np.arctan2(Iy,Ix)

	return H,theta

def nms(E,H,theta):
    #placeholder
    verticalRange = np.logical_or(np.logical_and(theta>=np.pi*3/8, theta<=np.pi*5/8),np.logical_and(theta>=np.pi*11/8,theta<=np.pi*13/8))
    horizontalRange = np.logical_or(np.logical_and(theta>=np.pi*15/8, theta<=np.pi/8),np.logical_and(theta>=np.pi*7/8,theta<=np.pi*9/8))
    digRange = np.logical_or(np.logical_and(theta>=np.pi*5/8,theta<=np.pi*7/8),np.logical_and(theta>=np.pi*13/8,theta<=np.pi*15/8))
    antiDigRange = np.logical_or(np.logical_and(theta>=np.pi/8,theta<=np.pi*3/8),np.logical_and(theta>=np.pi*9/8,theta<=np.pi*11/8))

    h = np.array([[0,0,0],[-1,1,-1],[0,0,0]])
    v = np.array([[0,-1,0],[0,1,0],[0,-1,0]])
    d = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    antiD = np.array([[0,0,-1],[0,1,0],[-1,0,0]])

    horizontalIndex = np.where(np.logical_and(horizontalRange,conv2(H,h,mode='same')<=0))
    E[horizontalIndex] = 0
    verticalIndex = np.where(np.logical_and(verticalRange,conv2(H,v,mode='same')<=0))
    E[verticalIndex] = 0
    digRangeIndex = np.where(np.logical_and(digRange,conv2(H,d,mode='same')<=0))
    E[digRangeIndex] = 0
    antiDigRangeIndex = np.where(np.logical_and(antiDigRange,conv2(H,antiD,mode='same')<=0))
    E[antiDigRangeIndex] = 0

    return E

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.jpg')))/255.

H,theta = grads(img)

imsave(fn('outputs/prob3_a.jpg'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.jpg'),E0)
imsave(fn('outputs/prob3_b_1.jpg'),E1)
imsave(fn('outputs/prob3_b_2.jpg'),E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.jpg'),E0n)
imsave(fn('outputs/prob3_b_nms1.jpg'),E1n)
imsave(fn('outputs/prob3_b_nms2.jpg'),E2n)
