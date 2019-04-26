## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i):
    # Placeholder
    exp_s = np.zeros((2*K+1, 2*K+1))
    h, w = X.shape[0], X.shape[1]

    for i in range(2*K+1):
        for j in range(2*K+1):
            exp_s[i][j] = np.exp(-(np.power(i-K,2)+np.power(j-K,2))/(2*sgm_s*sgm_s))

    sum_b = np.zeros((h,w,3))
    outputImg = np.zeros((h,w,3))
    for c in range(3):
        for x in range(-K, K+1):
            for y in range(-K, K+1):
                o_y1,o_y2,i_y1,i_y2 = max(0,y), min(h+y,h),max(-y,0), min(h,h-y)
                o_x1,o_x2,i_x1,i_x2 = max(0,x), min(w+x,w),max(-x,0), min(w,w-x)
                exp_i = X[i_y1:i_y2, i_x1:i_x2, c]-X[o_y1:o_y2, o_x1:o_x2, c]
                exp_i = -np.power(exp_i,2)/(2*sgm_i*sgm_i)
                exp_i = np.exp(exp_i)
                bxy = exp_i*exp_s[y+K][x+K]
                outputImg[o_y1:o_y2,o_x1:o_x2,c] += bxy * X[i_y1:i_y2,i_x1:i_x2,c]
                sum_b[o_y1:o_y2, o_x1:o_x2, c] += bxy
    outputImg = outputImg/sum_b
    

    return outputImg


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.jpg')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.jpg')))/255.

K=9

print("Creating outputs/prob4_1_a.jpg")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.jpg'),clip(im1A))


print("Creating outputs/prob4_1_b.jpg")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.jpg'),clip(im1B))

print("Creating outputs/prob4_1_c.jpg")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.jpg'),clip(im1C))

# Repeated application
print("Creating outputs/prob4_1_rep.jpg")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.jpg'),clip(im1D))

# Try this on image with more noise    
print("Creating outputs/prob4_2_rep.jpg")
im2D = bfilt(img2,2,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.jpg'),clip(im2D))
