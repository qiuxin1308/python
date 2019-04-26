## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):

	#hints from the lec11 slide
	px, py, ppx, ppy = pts[:,0], pts[:,1], pts[:,2], pts[:,3]
	px, py, ppx, ppy = px.reshape((-1,1)), py.reshape((-1,1)), ppx.reshape((-1,1)), ppy.reshape((-1,1))
	zeros, pz = np.zeros(px.shape), np.ones(px.shape)
	rowA1 = [zeros,zeros,zeros,-px,-py,-pz,ppy*px,ppy*py,ppy*pz]
	rowA2 = [px,py,pz,zeros,zeros,zeros,-ppx*px,-ppx*py,-ppx*pz]
	rowA3 = [-ppy*px,-ppy*py,-ppy*pz,ppx*px,ppx*py,ppx*pz,zeros,zeros,zeros]
	#concatenate
	row1 = np.concatenate(rowA1,axis=1)
	row2 = np.concatenate(rowA2,axis=1)
	row3 = np.concatenate(rowA3,axis=1)
	A = np.concatenate([row1,row2,row3])
	u,s,v = np.linalg.svd(A)
	H = v[-1].reshape((3,3))

	return H
    

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):
	h,w = src.shape[0], src.shape[1]
	spts = np.float32([[0,0],[0,w-1],[h-1,0],[h-1,w-1]])
	dpts = dpts.astype(int)
	H = getH(np.concatenate((dpts,spts),axis=1))
	x_min, x_max = min(dpts[:,0]), max(dpts[:,0])+1
	y_min, y_max = min(dpts[:,1]), max(dpts[:,1])+1
	for x in range(x_min, x_max):
		for y in range(y_min, y_max):
			p = np.asarray([x, y, 1]).transpose()
			hp = np.dot(H,p)
			hp[0] = hp[0] / hp[2]
			hp[1] = hp[1] / hp[2]
			if hp[1] > 0 and hp[1] < w - 1 and hp[0] > 0 and hp[0] < h - 1:
				#bi-linear iterpolation based on the lecture slide
				fx = abs(hp[0] - int(hp[0]))
				fy = abs(hp[1] - int(hp[1]))
				i, j = int(hp[0]), int(hp[1])
				d1 = fx * (fy * src[i,j] + (1 - fy) * src[i, j+1])
				d2 = (1 - fx) * (fy * src[i+1,j] + (1 - fy) * src[i+1,j+1])
				dest[y, x] = d1 + d2

	dest = np.where(dest > 1, 1, dest) #set to 1 if it is larger than 1

	return dest

    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/prob4.png'),comb)
