import sys
sys.path.insert(0, "/Users/anthonygruber/Desktop/Projects/AMv2/src")

import numpy as np
import numethod as nm
import basicFuncs as bf
import pandas as pd
import scipy as sy
from sklearn import preprocessing

# import os
# os.chdir('/Users/anthonygruber/Desktop/Projects/AMv2')
# npz = np.load('/Users/anthonygruber/Desktop/Projects/AMv2/data_MHD/kl-short.npz')
# df= pn.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index')
#
# print df.shape
# df[9990][1]
# np.asarray(map(lambda x: bf.Squaresum(*x), am)) - fvals
# len(mesh)

mesh, fSamples, paths, realgrads = nm.build_random_data( 5, 250, bf.Squaresum, bf.gradSquaresum)
nm.mainRandEx_old( mesh, fSamples, realgrads, 0.02, 50, nm.get_random_init_pt(5), 0, False)
nm.subspEx(mesh, fSamples, realgrads, 2, 0, .2)
am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(5), mesh, fSamples, realgrads, 0.005)
# nm.splinePlot( am, fvals,'squaresum','{ss}')

## Squaresum with 3 dims, 30^3 points, and 1/3 test fraction.
## Took 247 seconds for the paper, now takes ~4.
## Plot looks great, too.  I also made f(am) values more accurate.
mesh, fSamples, paths, realgrads = nm.build_data( 3, 30, bf.Squaresum, bf.gradSquaresum)
nm.mainRandEx_old( mesh, fSamples, realgrads, 0.05, 0.33, nm.get_random_init_pt(3))
nm.subspExOld( mesh, fSamples, realgrads, 2, 0, .33)
# am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(3), mesh, fSamples, realgrads, 0.005)
# nm.splinePlot( am, fvals, 'squaresum', '{ss}')

## 12 dim with 5k pts.  Not that accurate.
## Takes about ~10s
mesh, fSamples, paths, realgrads = nm.build_random_data( 12, 5000, bf.Squaresum, bf.gradSquaresum)
nm.mainRandEx_old( mesh, fSamples, realgrads, 0.1, 1000, nm.get_random_init_pt(12), 0)
nm.subspExOld( mesh, fSamples, realgrads, 1, 0, .2)
# am, fvals = nm.build_AM_from_data( nm.get_random_init_pt( 12, 34), mesh, fSamples, realgrads, 0.1)
# nm.splinePlot( am, fvals, 'squaresum', '{ss}')

## Remember that function f(x,y) = sin(8x) + sin(8y) with a ton of critical points?
## Absolutely no problem now.
mesh, fSamples, paths, realgrads = nm.build_random_data( 2, 1000, bf.uhoh, bf.graduhoh)
nm.mainRandEx_old( mesh, fSamples, realgrads, 0.01, 200, nm.get_random_init_pt(2), 0)
nm.subspExOld( mesh, fSamples, realgrads, 2, 0, .2)
# am,fvals = nm.build_AM_from_data( nm.get_random_init_pt(2), mesh, fSamples, paths, realgrads, 0.005)
# nm.splinePlot( am, fvals, 'lots of cps','l')

## Singularities are even 'ok' provided they are handled appropriately
## (want to think more about how to do this in general)
## This example is the argument function which technically requires a branch cut
## Plot is ugly because it is winding around the origin a million times
mesh, fSamples, paths, realgrads = nm.build_random_data( 2, 1000, bf.f4, bf.gradf4)
nm.mainRandEx_old( mesh, fSamples, realgrads, 0.1, 200, nm.get_random_init_pt(2), 0)
nm.subspExOld( mesh, fSamples, realgrads, 2, 0, .2)
# am,fvals = nm.build_AM_from_data( nm.get_random_init_pt(2), mesh, fSamples, paths, realgrads, 0.02)
# nm.splinePlot( am, fvals, 'argz', 'z')

## Can also do all the MHD stuff from the paper faster and more accurately.
## Takes maybe ~5s each (or less).
meshy = nm.make_mesh( 5, 0.2)
tree = sy.spatial.KDTree( meshy)
u, pathsu, rgu = bf.get_u_avg_data( meshy)
b, pathsb, rgb = bf.get_b_ind_data( meshy)

## Hartmann for u
nm.mainRandEx_old( meshy, u, rgu, 0.05, 0.2, nm.get_random_init_pt(5), 0)
nm.subspExOld( meshy, u, rgu, 3, 0, 0.2)
# am,fvals = nm.build_AM_from_data( nm.get_random_init_pt(5), meshy, u, pathsu, rgu, 0.01)
# nm.splinePlot( am, fvals, 'u_{avg}', '{Hu}')

## Hartmann for B
nm.mainRandEx_old( meshy, b, rgb, 0.05, 0.2, nm.get_random_init_pt(5), 0)
nm.subspExOld( meshy, b, rgb, 3, 0, .2)
# am,fvals = nm.build_AM_from_data( nm.get_random_init_pt(5), meshy, b, pathsb, rgb, 0.01)
# nm.splinePlot( am,fvals,'B_{ind}','{HB}')

## Setup for real data MHD
# Load data from data files
data = pd.read_csv('./data/MHD/MHD_Generator_Data_Uavg.txt').values
Xu = data[:,1:6]; u = data[:,6]; du = data[:,7:]

data = pd.read_csv('./data/MHD/MHD_Generator_Data_Bind.txt').values
XB = data[:,1:6]; B = data[:,6]; dB = data[:,7:]

# New upper/lower bounds
lb = np.log(np.array([.001, .1, .1, .1, .1]))
ub = np.log(np.array([.01, 10, .5, 10, 1]))

# Scale gradients according to the chain rule, get normalized inputs
realdu = .5*(ub - lb)*Xu*du; XXu = 2*(np.log(Xu) - lb)/(ub - lb) - 1
realdB = .5*(ub - lb)*XB*dB; XXB = 2*(np.log(XB) - lb)/(ub - lb) - 1

## MHD for u
nm.mainRandEx_old( XXu, u, realdu, 0.05, 97, nm.get_random_init_pt(5), 0)
nm.splinePlot( am, fvals, 'u_{avg}', '{u}' )
nm.subspEx( XXu, u, realdu, 2, 0, .2)
# nm.subspExOld( XXu, u, realdu, 1, 0, .2)
# am,fvals = nm.build_AM_from_data( nm.get_random_init_pt(5), XXu, u, realdu, 0.02)

## MHD for B
nm.mainRandEx_old( XXB, B, realdB, 0.05, 97, nm.get_random_init_pt(5), 0)
nm.subspEx( XXB, B, realdB, 2, 0, .2)
# am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(5), XXB, B, realdB, 0.02)
# nm.splinePlot( am, fvals, 'B_{ind}', '{B}')
