import os
os.chdir('/Users/anthonygruber/Desktop/Projects/AMv2')
import numpy as np
import pandas as pd
import numethod as nm
import basicFuncs as bf
import scipy as sy
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Liberian lower and upper parameter bounds
lb_L = np.array([.1, .1, .05, .41, .0276, .081, .25, .0833])
ub_L = np.array([.4, .4, .2, 1, .1702, .21, .5, .7])
#Sierra Leone bounds
lb_S = np.array([.1, .1, .05, .41, .0275, .1236, .25, .0833])
ub_S = np.array([.4, .4, .2, 1, .1569, .384, .5, .7])

#basic reproduction number
def R0(x):
    b1 = x[:,0]; b2 = x[:,1]; b3 = x[:,2]; r1 = x[:,3]
    g1 = x[:,4]; g2 = x[:,5]; om = x[:,6]; p = x[:,7]

    return (b1 + b2*r1*g1/om + b3*p/g2)/(g1 + p)

#reproduction number gradient with respect to UNnormalized parameters
def R0_grad(x):
    b1 = x[:,0]; b2 = x[:,1]; b3 = x[:,2]; r1 = x[:,3]
    g1 = x[:,4]; g2 = x[:,5]; om = x[:,6]; p = x[:,7]

    dRdb1 = (1./(g1 + p))[:,None]
    dRdb2 = (r1*g1/om/(g1 + p))[:,None]
    dRdb3 = (p/g2/(g1 + p))[:,None]
    dRdr1 = (b2*g1/om/(g1 + p))[:,None]
    dRdg1 = (b2*r1/om/(g1 + p) - R0(x)/(g1 + p))[:,None]
    dRdg2 = (-b3*p/g2**2/(g1 + p))[:,None]
    dRdom = (-b2*r1*g1/om**2/(g1 + p))[:,None]
    dRdp = (b3/g2/(g1 + p) - R0(x)/(g1 + p))[:,None]

    return np.hstack((dRdb1, dRdb2, dRdb3, dRdr1, dRdg1, dRdg2, dRdom, dRdp))

# make mesh
mesh = nm.build_random_data( 8, 2000, bf.Squaresum, bf.gradSquaresum)[0]

#un-normalized inputs for Liberia (S) and Sierra Leone (S)
x_L = lb_L + (ub_L - lb_L)/2.*( np.array( mesh) + 1)
x_S = lb_S + (ub_S - lb_S)/2.*( np.array( mesh) + 1)

#Their function values/gradient values
R_L = R0(x_L)
dR_L = R0_grad(x_L)
R_S = R0(x_S)
dR_S = R0_grad(x_S)

#Gradients with respect to normalized inputs according to the chain rule
dR_L *= (ub_L - lb_L)/2.
dR_S *= (ub_S - lb_S)/2.

#Unit-Length gradients w.r.t normalized inputs for AM
dR_L_path = preprocessing.normalize(dR_L)
dR_S_path = preprocessing.normalize(dR_S)


## Liberia ##

nm.mainRandEx_old( mesh, R_L, dR_L, 0.1, 400, nm.get_random_init_pt(8))
nm.subspEx(mesh, R_L, dR_L, 1, 0, .2)
nm.subspExOld(mesh, R_L, dR_L, 1, 0, .2)

#Build the AM for visualization
am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(8), mesh, R_L, dR_L_path, dR_L, 0.01)

#Fit R_0 vals along AM with splines and produce spline plot
nm.splinePlot( am, fvals,'Liberia','{L}')

#Values needed for other plots
numpts = len(am)
sValues = np.linspace(0., numpts, numpts) / (numpts)

#Coordinate Plot
fig=plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r'Coordinate Values of $\gamma_L(t)$', fontsize = 13)
ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
plt.plot(sValues,am[:,0], '-o', c='#66c2a5', label = r'$\beta_1$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,1], '-o', c='#fc8d62', label = r'$\beta_2$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,2], '-o', c='#8da0cb', label = r'$\beta_3$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,3], '-o', c='#e78ac3', label = r'$\rho_1$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,4], '-o', c='#a6d854', label = r'$\Gamma_1$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,5], '-o', c='#ffd92f', label = r'$\Gamma_2$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,6], '-o', c='#e5c494', label = r'$\omega$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,7], '-o', c='#b3b3b3', label = r'$\psi$', markersize=3, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlim(0,1)
plt.savefig('LiberiaaCoords.pdf',bbox_inches='tight')

#Derivative Plot
fig=plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r'Coordinate Derivatives Along $\gamma_L(t)$', fontsize = 13)
ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
plt.plot(sValues,np.abs(np.gradient(am[:,0])), '-o', c='#66c2a5', label = r'$|\beta_1^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,1])), '-o', c='#fc8d62', label = r'$|\beta_2^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,2])), '-o', c='#8da0cb', label = r'$|\beta_3^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,3])), '-o', c='#e78ac3', label = r'$|\rho_1^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,4])), '-o', c='#a6d854', label = r'$|\Gamma_1^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,5])), '-o', c='#ffd92f', label = r'$|\Gamma_2^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,6])), '-o', c='#e5c494', label = r'$|\omega^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,7])), '-o', c='#b3b3b3', label = r'$|\psi^\prime|$', markersize=3, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlim(0,1)
plt.savefig('liberiaDerivs.pdf', bbox_inches = 'tight')


## Sierra Leone ##

nm.mainRandEx_old( mesh, R_S, dR_S, 0.1, 400, nm.get_random_init_pt(8))
nm.subspEx( mesh, R_S, dR_S, 1, 0, .2)

#Build the AM for visualization
am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(8), mesh, R_S, dR_S_path, dR_S, 0.01)

#Fit R_0 vals along AM with splines and produce spline plot
nm.splinePlot( am, fvals,'Sierra-Leone','{S}')

#Values needed for other plots
numpts = len(am)
sValues = np.linspace(0., numpts, numpts) / (numpts)


#Coordinate Plot
fig=plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r'Coordinate Values of $\gamma_{SL}(t)$', fontsize = 13)
ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
plt.plot(sValues,am[:,0], '-o', c='#66c2a5', label = r'$\beta_1$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,1], '-o', c='#fc8d62', label = r'$\beta_2$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,2], '-o', c='#8da0cb', label = r'$\beta_3$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,3], '-o', c='#e78ac3', label = r'$\rho_1$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,4], '-o', c='#a6d854', label = r'$\Gamma_1$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,5], '-o', c='#ffd92f', label = r'$\Gamma_2$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,6], '-o', c='#e5c494', label = r'$\omega$', markersize=3, linewidth=2)
plt.plot(sValues,am[:,7], '-o', c='#b3b3b3', label = r'$\psi$', markersize=3, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlim(0,1)
plt.savefig('SLCoords.pdf',bbox_inches='tight')

#Derivative Plot
fig=plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r'Coordinate Derivatives Along $\gamma_{SL}(t)$', fontsize = 13)
ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
plt.plot(sValues,np.abs(np.gradient(am[:,0])), '-o', c='#66c2a5', label = r'$|\beta_1^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,1])), '-o', c='#fc8d62', label = r'$|\beta_2^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,2])), '-o', c='#8da0cb', label = r'$|\beta_3^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,3])), '-o', c='#e78ac3', label = r'$|\rho_1^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,4])), '-o', c='#a6d854', label = r'$|\Gamma_1^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,5])), '-o', c='#ffd92f', label = r'$|\Gamma_2^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,6])), '-o', c='#e5c494', label = r'$|\omega^\prime|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,7])), '-o', c='#b3b3b3', label = r'$|\psi^\prime|$', markersize=3, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlim(0,1)
plt.savefig('SLDerivs.pdf',bbox_inches='tight')


#os.chdir(os.path.join(os.path.expanduser("~"),'active-manifolds/code_anthony/src'))
