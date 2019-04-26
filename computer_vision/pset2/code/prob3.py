### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):
    #mean of the R, G and B values
    N,H,W = len(imgs), len(imgs[0]), len(imgs[0][0])
    I = np.zeros((N,H,W))
    for i in range(N):
        I[i,:,:] = (imgs[i][:,:,0] + imgs[i][:,:,1] + imgs[i][:,:,2])/3
    I = I.reshape((N,H*W)) 
    #calculate n
    n = np.linalg.solve(L.transpose().dot(L),L.transpose().dot(I))
    n = np.asarray(n.transpose())
    n = n.reshape((len(mask),len(mask[0]),3))
    #normalize
    nml = np.sqrt(n[:,:,0]*n[:,:,0]+n[:,:,1]*n[:,:,1]+n[:,:,2]*n[:,:,2])
    n[:,:,0] = np.where(mask>0, n[:,:,0]/nml,0)
    n[:,:,1] = np.where(mask>0, n[:,:,1]/nml,0)
    n[:,:,2] = np.where(mask>0, n[:,:,2]/nml,0)

    return n


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):
    alb = np.zeros(imgs[0].shape)
    I = np.zeros((len(nrm),len(nrm[0]),len(imgs),3))
    for i in range(len(L)):
        I[:,:,i,:] = imgs[i]
    dem = np.sum(np.power(nrm.dot(L.transpose()),2),axis=2)
    num = np.zeros(imgs[0].shape)
    num[:,:,0] = np.sum(nrm.dot(L.transpose())*I[:,:,:,0],axis=2)
    num[:,:,1] = np.sum(nrm.dot(L.transpose())*I[:,:,:,1],axis=2)
    num[:,:,2] = np.sum(nrm.dot(L.transpose())*I[:,:,:,2],axis=2)
    alb[:,:,0] = np.where(mask>0, num[:,:,0]/dem, 0)
    alb[:,:,1] = np.where(mask>0, num[:,:,1]/dem, 0)
    alb[:,:,2] = np.where(mask>0, num[:,:,2]/dem, 0)

    return alb
    
########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
