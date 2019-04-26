## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

## Take color image, and return 'white balanced' color image
## based on gray world, as described in Problem 2(a). For each
## channel, find the average intensity across all pixels.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2a(img):
	sizeOfColor = np.size(img) / 3
	r = np.sum(img[:,:,0]) / sizeOfColor
	g = np.sum(img[:,:,1]) / sizeOfColor
	b = np.sum(img[:,:,2]) / sizeOfColor
	rgb = np.zeros(3)
	rgb[0], rgb[1], rgb[2] = r, g, b
	rgb_n = rgb / np.sum(rgb) * 3
	img[:,:,0] = img[:,:,0] / rgb_n[0]
	img[:,:,1] = img[:,:,1] / rgb_n[1]
	img[:,:,2] = img[:,:,2] / rgb_n[2]

	return img


## Take color image, and return 'white balanced' color image
## based on description in Problem 2(b). In each channel, find
## top 10% of the brightest intensities, take their average.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2b(img):
	sizeOfColor = np.size(img) / 3
	r,g,b = np.sort(img[:,:,0],axis=None), np.sort(img[:,:,1],axis=None), np.sort(img[:,:,2],axis=None)
	rgb = np.zeros(3)
	h = int(0.9*sizeOfColor)
	r,g,b = r[h:],g[h:],b[h:]
	rgb[0], rgb[1], rgb[2] = 1/np.mean(r), 1/np.mean(g), 1/np.mean(b)
	rgb = rgb / np.sum(rgb) * 3
	img[:,:,0] = img[:,:,0] * rgb[0]
	img[:,:,1] = img[:,:,1] * rgb[1]
	img[:,:,2] = img[:,:,2] * rgb[2]

	return img



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program
im1 = np.float32(imread(fn('inputs/CC/ex1.jpg')))/255.
im2 = np.float32(imread(fn('inputs/CC/ex2.jpg')))/255.
im3 = np.float32(imread(fn('inputs/CC/ex3.jpg')))/255.


im1a = balance2a(im1)
im2a = balance2a(im2)
im3a = balance2a(im3)

imsave(fn('outputs/prob2a_1.png'),clip(im1a))
imsave(fn('outputs/prob2a_2.png'),clip(im2a))
imsave(fn('outputs/prob2a_3.png'),clip(im3a))

im1b = balance2b(im1)
im2b = balance2b(im2)
im3b = balance2b(im3)

imsave(fn('outputs/prob2b_1.png'),clip(im1b))
imsave(fn('outputs/prob2b_2.png'),clip(im2b))
imsave(fn('outputs/prob2b_3.png'),clip(im3b))
