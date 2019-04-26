## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################





## Fill out these functions yourself

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):
    W = img.shape[1]
    H = img.shape[0]
    c = np.zeros([H,W],dtype=np.uint32)
    j = 0

    # hints from the solution key 1
    for y in range(-2,3):
        for x in range(-2,3):
            if y < 0:
                y1a, y1b, y2a, y2b = 0, -y, H+y, H
            else:
                y1a, y1b, y2a, y2b = y, 0, H, H-y
            if x < 0:
                x1a, x1b, x2a, x2b = 0, -x, W+x, W
            else:
                x1a, x1b, x2a, x2b = x, 0, W, W-x

            intensity = np.uint32(np.power(2, j))
            cxy = intensity * (img[y1b:y2b,x1b:x2b] > img[y1a:y2a,x1a:x2a])
            c[y1b:y2b,x1b:x2b] = c[y1b:y2b,x1b:x2b] + cxy
            j = j + 1

    return c
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):
    left_c = census(left)
    right_c = census(right)
    H,W,D = left_c.shape[0], left_c.shape[1], dmax+1
    d = np.ones(left_c.shape)

    #generate 3D matrix W*H*D
    possible_inf = float('inf')
    matrix = np.ones((H,W,D)) * possible_inf
    matrix[:, :, 0] = hamdist(left_c, right_c)
    for i in range(1,D):
        matrix[:,i:,i] = hamdist(left_c[:,i:], right_c[:,0:(W-i)])
    best = np.amin(matrix, axis=2)

    for i in range(1,D):
        d = np.where(np.logical_and(matrix[:,:,i] == best, d==1), i, d)
    d = np.where(d==1,0,d)

    return d
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
