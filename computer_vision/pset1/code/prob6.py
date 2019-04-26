## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

def im2wv(img,nLev):
    # Placeholder that does nothing
    outputImg = []
    outputImg.append(img)
    h = img.shape[0]
    w = img.shape[1]
    harrFactor = 0.5 * np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    if nLev == 0:
        return outputImg
    for i in range(nLev):
        lastElement = outputImg[-1]
        del outputImg[-1]
        a = lastElement[0::2,0::2]
        b = lastElement[0::2,1::2]
        c = lastElement[1::2,0::2]
        d = lastElement[1::2,1::2]
        abcdMatrix = [a,b,c,d]
        L = np.zeros((h//np.power(2,i+1),w//np.power(2,i+1)))
        H1 = np.zeros((h//np.power(2,i+1),w//np.power(2,i+1)))
        H2 = np.zeros((h//np.power(2,i+1),w//np.power(2,i+1)))
        H3 = np.zeros((h//np.power(2,i+1),w//np.power(2,i+1)))
        L = harrFactor[0][0] * abcdMatrix[0] + harrFactor[0][1] * abcdMatrix[1] + harrFactor[0][2] * abcdMatrix[2] + harrFactor[0][3] * abcdMatrix[3]
        H1 = harrFactor[1][0] * abcdMatrix[0] + harrFactor[1][1] * abcdMatrix[1] + harrFactor[1][2] * abcdMatrix[2] + harrFactor[1][3] * abcdMatrix[3]
        H2 = harrFactor[2][0] * abcdMatrix[0] + harrFactor[2][1] * abcdMatrix[1] + harrFactor[2][2] * abcdMatrix[2] + harrFactor[2][3] * abcdMatrix[3]
        H3 = harrFactor[3][0] * abcdMatrix[0] + harrFactor[3][1] * abcdMatrix[1] + harrFactor[3][2] * abcdMatrix[2] + harrFactor[3][3] * abcdMatrix[3]
        outputImg.append([H1,H2,H3])
        outputImg.append(L)

    return outputImg


def wv2im(pyr):
    # Placeholder that does nothing
    harrFactor = 0.5 * np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    inverseHarrFactor = np.linalg.inv(harrFactor)
    #print(inverseHarrFactor)
    nLev = len(pyr) - 1
    for i in range(nLev):
        L = pyr[-1]
        listOfH = pyr[-2]
        h,w = L.shape[0],L.shape[1]
        H1,H2,H3 = listOfH[0],listOfH[1],listOfH[2]
        curLevMatrix = [L,H1,H2,H3]
        outputImg = np.zeros((h*2,w*2))
        a = inverseHarrFactor[0][0] * curLevMatrix[0] + inverseHarrFactor[0][1] * curLevMatrix[1] + inverseHarrFactor[0][2] * curLevMatrix[2] + inverseHarrFactor[0][3] * curLevMatrix[3]
        b = inverseHarrFactor[1][0] * curLevMatrix[0] + inverseHarrFactor[1][1] * curLevMatrix[1] + inverseHarrFactor[1][2] * curLevMatrix[2] + inverseHarrFactor[1][3] * curLevMatrix[3]
        c = inverseHarrFactor[2][0] * curLevMatrix[0] + inverseHarrFactor[2][1] * curLevMatrix[1] + inverseHarrFactor[2][2] * curLevMatrix[2] + inverseHarrFactor[2][3] * curLevMatrix[3]
        d = inverseHarrFactor[3][0] * curLevMatrix[0] + inverseHarrFactor[3][1] * curLevMatrix[1] + inverseHarrFactor[3][2] * curLevMatrix[2] + inverseHarrFactor[3][3] * curLevMatrix[3]
        outputImg[0::2,0::2] = a
        outputImg[0::2,1::2] = b
        outputImg[1::2,0::2] = c
        outputImg[1::2,1::2] = d
        pyr = pyr[:-2]
        pyr.append(outputImg)

    return pyr[-1]



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.jpg')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.jpg'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.jpg'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.jpg'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.jpg'),im)

# Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.jpg' % i),im)
