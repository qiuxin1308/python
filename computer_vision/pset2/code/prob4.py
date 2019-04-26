## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement in Fourier Domain / Frankot-Chellappa
def ntod(nrm, mask, lmda):
	fx = np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]])
	fy = np.array([[0,-0.5,0],[0,0,0],[0,0.5,0]])
	fr = np.array([[-1/9,-1/9,-1/9],[-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]])
	nx,ny,nz = nrm[:,:,0],nrm[:,:,1],nrm[:,:,2]
	gx = np.where(mask>0,-1*(nx/nz),0)
	gy = np.where(mask>0,-1*(ny/nz),0)
	Gx,Gy = np.fft.fft2(gx),np.fft.fft2(gy)
	#should pad the kernel before fourier transform, hints from the pset1 solution key
	fx_p,fy_p,fr_p = np.zeros(mask.shape),np.zeros(mask.shape),np.zeros(mask.shape)
	fx_p[:2,:2],fx_p[:2,-1:],fx_p[-1:,:2],fx_p[-1:,-1:] = fx[1:,1:],fx[1:,:1],fx[:1,1:],fx[:1,:1]
	fy_p[:2,:2],fy_p[:2,-1:],fy_p[-1:,:2],fy_p[-1:,-1:] = fy[1:,1:],fy[1:,:1],fy[:1,1:],fy[:1,:1]
	fr_p[:2,:2],fr_p[:2,-1:],fr_p[-1:,:2],fr_p[-1:,-1:] = fr[1:,1:],fr[1:,:1],fr[:1,1:],fr[:1,:1]
	Fr,Fx,Fy = np.fft.fft2(fr_p),np.fft.fft2(fx_p),np.fft.fft2(fy_p)
	Fx2 = np.power(np.absolute(Fx),2)
	Fy2 = np.power(np.absolute(Fy),2)
	Fr2 = np.power(np.absolute(Fr),2)
	dem = Fx2 + Fy2 + lmda * Fr2 + 1e-12
	num = np.conj(Fx) * Gx + np.conj(Fy) * Gy
	fz = num / dem
	Fz = np.fft.ifft2(fz)
	Z = np.real(Fz)
	Z = np.where(mask>0, Z, 0)

	return Z

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-6)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
