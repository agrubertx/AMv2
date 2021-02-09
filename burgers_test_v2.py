import sys
sys.path.insert(0, "/Users/anthonygruber/Desktop/Projects/AMv2/")

import numpy as np
from sklearn import preprocessing
import src.numethod as nm
import src.burgers as bg
import active_subspaces as ac
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# number of Samples (Use 6 or 7, using 6 omits the 1000 point data set)
num = 5

# Initial mesh data
points = [0, 0, 0, 0, 0, 0, 0]
np.random.seed(0) # for reproducibility
points[6] = np.random.uniform( -1, 1, (1000, 3) )
points[5] = np.random.uniform( -1, 1, (500, 3) )
points[4] = np.random.uniform( -1, 1, (200, 3) )
points[3] = np.random.uniform( -1, 1, (100, 3) )
points[2] = np.random.uniform( -1, 1, (50, 3) )
points[1] = np.random.uniform( -1, 1, (25, 3) )
points[0] = np.random.uniform( -1, 1, (10, 3) )

# Upper and Lower bounds for parameters, taken from Carlberg paper
UBs = np.array( [45, 5.5, 0.03], dtype=float )
LBs = np.array( [25, 4.25, 0.015], dtype=float )

# Linear scaling to appropriate range
nuPoints = np.zeros_like(points)

for i in range(num):
  nuPoints[i] = LBs + 0.5 * (UBs - LBs) * (1 + points[i])

# Step size for FD
# h = 0.0000001
# Q values and FD-generated gradQ values for these parameters -- outdated
# dataFD200 = map( lambda x: bg.generate_fd_gradients( h, nuSample200[x,0], nuSample200[x,1:] ),
#     range( len(nuSample200) ) )

# Sample Q and gradQ values over parameter space
data = np.zeros_like(points)

for i in range(num):
  data[i] = list ( map( lambda x: bg.burgers( 2, 256, 2000, nuPoints[i][x, 0], 1, nuPoints[i][x, 1:],
    False)[2:], range( len(nuPoints[i]) ) ) )

# Separate data appropriately
Qtwo = np.zeros_like(points)
Qthr = np.zeros_like(points)
QthrA = np.zeros_like(points)
gradQtwo = [ [ 0 for col in range(3) ] for row in range( len(points) ) ]
gradQthrA = [ [ 0 for col in range(3) ] for row in range( len(points) ) ]
gradQthr = [ [ 0 for col in range(3) ] for row in range( len(points) ) ]

for i in range(num):
  Qtwo[i] = np.array( [ data[i][j][0] for j in range( len(data[i]) ) ] )
  gradQtwo[i] = [ data[i][j][1] for j in range( len(data[i]) ) ]
  QthrA[i] = np.array( [ data[i][j][2] for j in range( len(data[i]) ) ] )
  gradQthrA[i] = [ data[i][j][3] for j in range( len(data[i]) ) ]
#  Qthr[i] = np.array( [ data[i][j][4] for j in range( len(data[i]) ) ] )
#  gradQthr[i] = [ data[i][j][5] for j in range( len(data[i]) ) ]

# Chain rule to get appropriate gradients in [-1,1]^3
nuGradsTwo = [ [ 0 for col in range(3) ] for row in range( len(points) ) ]
nuGradsThrA = [ [ 0 for col in range(3) ] for row in range( len(points) ) ]
# nuGradsThr = [ [ 0 for col in range(3) ] for row in range( len(points) ) ]

for i in range(num):
  nuGradsTwo[i] = 0.5 * (UBs - LBs) * gradQtwo[i]
  nuGradsThrA[i] = 0.5 * (UBs - LBs) * gradQthrA[i]
#  nuGradsThr[i] = 0.5 * (UBs - LBs) * gradQthr[i]

# Main routine comparing AM and AS -- first on Q2
for i in range(num):
    nm.manifoldEx( points[i], Qtwo[i], nuGradsTwo[i], 0.1, 0.2, nm.get_random_init_pt(3, 0), 0, False,"/Users/anthonygruber/Desktop/Projects/AMv2/test.txt" )
    nm.subspEx(points[i], Qtwo[i], nuGradsTwo[i], 2, 0, 0.2, False)
    print(" ")

# Visualize the AM and AS for one set -- Q2
nm.manifoldEx( points[5], Qtwo[5], nuGradsTwo[5], 0.05, 0.2, nm.get_random_init_pt(3, 0), 0 )
nm.subspEx( points[5], Qtwo[5], nuGradsTwo[5], 1, 0, 0.2, True, True )

# 3D scatter
am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(3, 0), points[3], Qtwo[3], nuGradsTwo[3], 0.05)
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(points[3][:,0], points[3][:,1], points[3][:,2], color = "green")
ax.plot(am[:,0], am[:,1], am[:,2], color = "blue", linestyle = "-")
plt.title("simple 3D scatter plot")

plt.show()


# Derivative plot
am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(3, 0), points[5], Qtwo[5], nuGradsTwo[5], 0.05)
numpts = len(am)
sValues = np.linspace(0., numpts, numpts) / numpts
fig=plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r'Coordinate Derivatives Along $\gamma_2(t)$', fontsize = 13)
ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
plt.plot(sValues, np.abs(np.gradient(am[:,0])), '-o', c='#66c2a5', label = r'$|Q_T|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,1])), '-o', c='#fc8d62', label = r'$|Q_{\mu1}|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,2])), '-o', c='#8da0cb', label = r'$|Q_{\mu2}|$', markersize=3, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlim(0,1)
# plt.savefig('Q2derivs.pdf', bbox_inches = 'tight')

# Main routine comparing AM and AS -- now on Q3alt
for i in range(num):
    nm.manifoldEx( points[i], QthrA[i], nuGradsThrA[i], 0.1, 0.2, nm.get_random_init_pt(3, 0), 0, False )
    nm.subspEx( points[i], QthrA[i], nuGradsThrA[i], 2, 0, 0.2, False )

# Visualize the AM and AS for one set -- Q3alt
nm.manifoldEx( points[5], QthrA[5], nuGradsThrA[5], 0.05, 0.2, nm.get_random_init_pt(3, 0), 0 )
nm.subspEx( points[5], QthrA[5], nuGradsThrA[5], 1, 0, 0.2 )

am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(3, 0), points[5], QthrA[5], nuGradsThrA[5], 0.05)
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(nuPoints[5][:,0], nuPoints[5][:,1], nuPoints[5][:,2], color = "green")
ax.plot(am[:,0], am[:,1], am[:,2], color = "blue", linestyle = "-")
plt.title("simple 3D scatter plot")

plt.show()

# Derivative plot
am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(3, 0), points[5], QthrA[5], nuGradsThrA[5], 0.05)
numpts = len(am)
sValues = np.linspace(0., numpts, numpts) / numpts
fig=plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r'Coordinate Derivatives Along $\gamma_3(t)$', fontsize = 13)
ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
plt.plot(sValues,np.abs(np.gradient(am[:,0])), '-o', c='#66c2a5', label = r'$|Q_T|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,1])), '-o', c='#fc8d62', label = r'$|Q_{\mu1}|$', markersize=3, linewidth=2)
plt.plot(sValues,np.abs(np.gradient(am[:,2])), '-o', c='#8da0cb', label = r'$|Q_{\mu2}|$', markersize=3, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlim(0,1)
# plt.savefig('Q3derivs.pdf', bbox_inches = 'tight')






# Because this code is still running Python 2...
from __future__ import division

# Adjust simulation parameters -- a much coarser discretization works here
num = 4
numx = 60
numt = 180

# Zero data vector
data = np.zeros_like(points)

# Compute PDE solution and derivatives. Note that the final time is 35 in every case.
for i in range(num):
  data[i] = list ( map( lambda x: bg.burgers( 2, numx, numt, 35, 1, nuPoints[i][x, 1:],
                                             False)[0], range( len(nuPoints[i]) ) ) )

# Define x and t points, same for every number of samples.
xx = np.linspace(0, 100, numx)
dx = (100-0) / numx
dt = 35 / numt
tt = np.array([dt * i for i in range(numt+1)])

# Compute points in (x, t, \mu) space.
pointz = np.zeros_like(points)

for i in range(num):
  pointz[i] = np.array( [ np.concatenate( [[xx[j], tt[k]], nuPoints[i][l, 1:]] )
                         for j in range(len(xx)) for k in range(len(tt))
                         for l in range(len(nuPoints[i])) ] )

# Initialize all the stuff we need: scaled points, function, derivatives, scaled derivatives
nuPointz = np.zeros_like(pointz)
u = np.zeros_like(points)
dudx = np.zeros_like(points)
dudt = np.zeros_like(points)
dudmu1 = np.zeros_like(points)
dudmu2 = np.zeros_like(points)
gradu = np.zeros_like(points)
nuGradu = np.zeros_like(points)

# Ranges for affine rescaling (just the inverse of the function before)
UBs = np.array( [100, 35, 5.5, 0.03], dtype=float )
LBs = np.array( [0, 0, 4.25, 0.015], dtype=float )

# Collection of necessary quantities.  Note the order of the "for" loops.
for i in range(num):
  nuPointz[i] = 2 / (UBs - LBs) * (pointz[i] - LBs) - 1  # Application of rescaling function

  u[i] = np.array( [ data[i][l][0][k][j] for j in range(len(xx)) for k in range(len(tt))
                    for l in range(len(points[i])) ] )  # Function value

  dudx[i] = np.array( [ data[i][l][4][k][j] for j in range(len(xx)) for k in range(len(tt))
                       for l in range(len(points[i])) ] )  # Derivative in x

  dudmu1[i] = np.array( [data[i][l][1][k][j] for j in range(len(xx)) for k in range(len(tt))
                         for l in range(len(points[i]))] )  # Derivative in mu1

  dudmu2[i] = np.array( [data[i][l][2][k][j] for j in range(len(xx)) for k in range(len(tt))
                         for l in range(len(points[i]))] )  # Derivative in mu2

  dudt[i] = np.array( [data[i][l][3][k][j] for j in range(len(xx)) for k in range(len(tt))
                       for l in range(len(points[i]))] )  # Derivative in t

  gradu[i] = np.array( [ [dudx[i][j], dudt[i][j], dudmu1[i][j], dudmu2[i][j]]
                        for j in range(len(u[i])) ] )  # Gradient in (x, t, \mu)

  nuGradu[i] = 0.5 * (UBs - LBs) * gradu[i]  # Chain rule


# Choose some points in the training set pointz[3]
import scipy as sy
tree = sy.spatial.KDTree(nuPointz[3], 3000)

# Choose some parameter configuration not in training set.
# Assemble points where solution values are needed.
test1 = np.array( [ np.concatenate( [[xx[j], tt[k]], [4.3, 0.021]] )
                  for k in range(len(tt)) for j in range(len(xx)) ] )

test2 = np.array( [ np.concatenate( [[xx[j], tt[k]], [5.15, 0.0285]] )
                  for k in range(len(tt)) for j in range(len(xx)) ] )

# Map inputs to [-1,1]^4
nuTest1 = 2 / (UBs - LBs) * (test1 - LBs) - 1
nuTest2 = 2 / (UBs - LBs) * (test2 - LBs) - 1

# Build a KD-tree corresponding to the training points.
tree = sy.spatial.KDTree(nuPointz[1], 1000)

vals1 = [0,0,0,0]
vals2 = [0,0,0,0]

for i in range(4):
  idx = int(numx / 10 * numt)
  vals1[i] = np.array( [ nm.get_f_value( nuTest1[idx*(i+1) + j],
                      tree, u[1], nuGradu[1] ) for j in range(numx) ] )

  vals2[i] = np.array( [ nm.get_f_value( nuTest2[idx*(i+1) + j],
                      tree, u[1], nuGradu[1] ) for j in range(numx) ] )




 ii = [int(0.1 * numt * (j+1)) for j in range(4)]
ii
reload(bg)
U1 = bg.burgers_witheps( 2, numx, numt, 35, 1, [4.3, 0.021], 0.001, 1)[0]
U1
print(U1)
U2 = bg.burgers( 2, numx, numt, 35, 1, [5.15, 0.0285], False)[0]

ii = [int(0.1 * numt * (j+1)) for j in range(4)]
print(ii)

fig1 = plt.figure(figsize = (8, 6))
ax = fig1.add_subplot(111, label = 'first')
plt.plot(test1[0:numx][:,0], vals1[3], '--', c='green', label = 't=14', markersize=3, linewidth=3)
plt.plot(test1[0:numx][:,0], U1[0][ii[3]], '-', c='palegreen', label = 't=14 FOM', markersize=2, linewidth=2 )
plt.plot(test1[0:numx][:,0], vals1[2], '--', c='darkorange', label = 't=10.5', markersize=3, linewidth=3)
plt.plot(test1[0:numx][:,0], U1[0][ii[2]], '-', c='lightsalmon', label = 't=10.5 FOM', markersize=2, linewidth=2)
plt.plot(test1[0:numx][:,0], vals1[1], '--', c='darkblue', label = 't=7', markersize=3, linewidth=3)
plt.plot(test1[0:numx][:,0], U1[0][ii[1]], '-', c='skyblue', label = 't=7 FOM', markersize=2, linewidth=2)
plt.plot(test1[0:numx][:,0], vals1[0], '--', c='indigo', label = 't = 3.5', markersize=3, linewidth=3)
plt.plot(test1[0:numx][:,0], U1[0][ii[0]], '--', c='magenta', label = 't = 3.5 FOM', markersize=2, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

fig2 = plt.figure(figsize = (8, 6))
ax = fig2.add_subplot(111, label = 'second')
plt.plot(test2[0:numx][:,0], vals2[3], '--', c='green', label = 't=14', markersize=3, linewidth=3)
plt.plot(test2[0:numx][:,0], U2[0][ii[3]], '-', c='palegreen', label = 't=14 FOM', markersize=2, linewidth=2 )
plt.plot(test2[0:numx][:,0], vals2[2], '--', c='darkorange', label = 't=10.5', markersize=3, linewidth=3)
plt.plot(test2[0:numx][:,0], U2[0][ii[2]], '-', c='lightsalmon', label = 't=10.5 FOM', markersize=2, linewidth=2)
plt.plot(test2[0:numx][:,0], vals2[1], '--', c='darkblue', label = 't=7', markersize=3, linewidth=3)
plt.plot(test2[0:numx][:,0], U2[0][ii[1]], '-', c='skyblue', label = 't=7 FOM', markersize=2, linewidth=2)
plt.plot(test2[0:numx][:,0], vals2[0], '--', c='indigo', label = 't = 3.5', markersize=3, linewidth=3)
plt.plot(test2[0:numx][:,0], U2[0][ii[0]], '--', c='magenta', label = 't = 3.5 FOM', markersize=2, linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

mu1 = np.array([4.44798934e+00, 1.51335924e-02])
mu2 = np.array([5.10220733e+00, 2.06308383e-02])
mu3 = np.array([4.26925806e+00, 2.08011350e-02])

Utrain1, DsavTrain1 = bg.burgers(2, numx, numt, 35, 1, mu1, False)[:2]
Utrain2, DsavTrain2 = bg.burgers(2, numx, numt, 35, 1, mu2, False)[:2]
Utrain3, DsavTrain3 = bg.burgers(2, numx, numt, 35, 1, mu3, False)[:2]

# We interpolate between Utrain2 and Utrain1
# Compute true and interpolated values (using line parameter = 1/4)
interpmu = 0.75 * mu2 + (1-0.75) * mu3
trueU = bg.burgers(2, numx, numt, 35, 1, interpmu, False)[0]
interpU = [ 0.75 * Utrain2[0][i] + (1-0.75) * Utrain3[0][i] for i in range(len(Utrain2[0])) ]

# Approximate values from AM
test = np.array( [ np.concatenate( [[xx[j], tt[k]], interpmu] )
                  for k in range(len(tt)) for j in range(len(xx)) ] )

nuTest = 2 / (UBs - LBs) * (test - LBs) - 1

vals = [0,0,0,0]
test[0:numx]
test[500]
reload(nm)
nm.test_local_linear(nuPointz[3], u[3], nuGradu[3], 60, 180, mu1, mu3, 0.2, 9)

reload(nm)
vals = np.array( [[0 for i in range(4)] for j in range(numx)] )



###### ALL THIS IS GARBAGE #######

idx = int(numx / 10 * numt)
test[980]
vals = np.zeros(4, dtype=object)
# for i in range(4):
#   vals[i] = np.array( [ nm.get_f_value( nuTest[1280*(i+1) + j], tree, u[3], nuGradu[3] ) for j in range(80) ] )
reload(nm)
for i in range(4):
  vals[i] = np.array( [ nm.get_f_value( nuTest[10*numt*(i+1) + j], tree, u[3], nuGradu[3] ) for j in range(100) ] )

print("Errors in AM approximation")
for i in range(4):
  # k = 16*(i+1)
  k = int(0.1 * numt * (i+1))
  relErrorL1 = ( np.sum( np.abs( vals[i] - trueU[0][k]) ) / np.sum( np.abs(trueU[0][k]) ) * 100 )
  print("The relative L1 error for t = %f is %f" %( 3.5*(i+1), relErrorL1 ) )
  relErrorL2 = ( np.linalg.norm( vals[i] - trueU[0][k] ) / np.linalg.norm(trueU[0][k]) * 100 )
  print("The relative L2 error for t = %f is %f" % ( 3.5*(i+1), relErrorL2 ) )


for i in range(4):
  k = 25*(i+1)
  relErrorL1 = ( np.sum( np.abs( interpU[k] - trueU[0][k]) ) / np.sum( np.abs(trueU[0][k]) ) * 100 )
  print("The relative L1 error for t = %f is %f" %( 3.5*(i+1), relErrorL1 ) )
  relErrorL2 = ( np.linalg.norm( interpU[k] - trueU[0][k] ) / np.linalg.norm(trueU[0][k]) * 100 )
  print("The relative L2 error for t = %f is %f" % ( 3.5*(i+1), relErrorL2 ) )

# Errors for AM
print("Errors in AM approximation")
for i in [y+1 for y in range(4)]:
  relErrorL1 = ( np.sum( np.abs( vals[500*i:500*i+50] - trueU[0][10*i]) )
               / np.sum( np.abs(trueU[0][10*i]) ) * 100 )
  print("The relative L1 error for t = %f is %f" %( 3.5*i, relErrorL1 ) )
  relErrorL2 = ( np.linalg.norm( vals[500*i:500*i+50] - trueU[0][10*i] )
               / np.linalg.norm(trueU[0][10*i]) * 100 )
  print("The relative L2 error for t = %f is %f" % ( 3.5*i, relErrorL2 ) )

print(" ")

# Errors for linear interpolation
print("Errors in local linear approximation")
for i in [y+1 for y in range(4)]:
  relErrorL1 = ( np.sum( np.abs( interpU[10*i] - trueU[0][10*i]) )
               / np.sum( np.abs(trueU[0][10*i]) ) * 100 )
  print("The relative L1 error for t = %f is %f" %( 3.5*i, relErrorL1 ) )
  relErrorL2 = ( np.linalg.norm( interpU[10*i] - trueU[0][10*i] )
               / np.linalg.norm(trueU[0][10*i]) * 100 )
  print("The relative L2 error for t = %f is %f" % ( 3.5*i, relErrorL2 ) )


# Errors for AM
print("Errors in AM approximation")
for i in [y+1 for y in range(4)]:
  relErrorL1 = ( np.sum( np.abs( vals[500*i:500*i+50] - trueU[0][10*i]) )
               / np.sum( np.abs(trueU[0][10*i]) ) * 100 )
  print("The relative L1 error for t = %f is %f" %( 3.5*i, relErrorL1 ) )
  relErrorL2 = ( np.linalg.norm( vals[500*i:500*i+50] - trueU[0][10*i] )
               / np.linalg.norm(trueU[0][10*i]) * 100 )
  print("The relative L2 error for t = %f is %f" % ( 3.5*i, relErrorL2 ) )

print(" ")

plt.scatter(pointz[0][:,2], pointz[0][:, 3])

# Errors for local linear interpolation
print("Errors in local linear approximation")
for i in [y+1 for y in range(4)]:
  relErrorL1 = ( np.sum( np.abs( interpU[10*i] - trueU[0][10*i]) )
               / np.sum( np.abs(trueU[0][10*i]) ) * 100 )
  print("The relative L1 error for t = %f is %f" %( 3.5*i, relErrorL1 ) )
  relErrorL2 = ( np.linalg.norm( interpU[10*i] - trueU[0][10*i] )
               / np.linalg.norm(trueU[0][10*i]) * 100 )
  print("The relative L2 error for t = %f is %f" % ( 3.5*i, relErrorL2 ) )



  xx = np.linspace(0, 100, numx)
  dx = (100-0) / numx
  dt = 35 / numt
  tt = np.array([dt * i for i in range(numt+1)])

  UBs = np.array( [100, 35, 5.5, 0.03], dtype=float )
  LBs = np.array( [0, 0, 4.25, 0.015], dtype=float )

  Utrain1 = bg.burgers(2, numx, numt, 35, 1, mu1, False)[0]
  Utrain2 = bg.burgers(2, numx, numt, 35, 1, mu2, False)[0]

  tree = sy.spatial.KDTree(nuPointz[3], 3000)

  interpmu = 0.75 * mu2 + (1-0.75) * mu3

  trueU = bg.burgers(2, numx, numt, 35, 1, interpmu, False)[0]

  interpU = [ 0.75 * Utrain2[0][i] + (1-0.75) * Utrain1[0][i]
              for i in range(len(Utrain1[0])) ]

  test = np.array( [ np.concatenate( [[xx[j], tt[k]], interpmu] )
                  for k in range(len(tt)) for j in range(len(xx)) ] )

  nuTest = 2 / (UBs - LBs) * (test - LBs) - 1


  vals = np.zeros(4, dtype=object)

  for i in range(4):
    print(10*numt*(i+1))
    vals[i] = np.array( [ nm.get_f_value( nuTest[10*numt*(i+1) + j],
                       tree, u[3], nuGradu[3] ) for j in range(numx) ] )

  print("Errors in AM approximation")
  for i in range(4):
    # k = 16*(i+1)
    k = int(0.1 * numt * (i+1))
    relErrorL1 = ( np.sum( np.abs( vals[i] - trueU[0][k]) )
                  / np.sum( np.abs(trueU[0][k]) ) * 100 )
    print("The relative L1 error for t = %f is %f" %( 3.5*(i+1), relErrorL1 ))
    relErrorL2 = ( np.linalg.norm( vals[i] - trueU[0][k] )
                  / np.linalg.norm(trueU[0][k]) * 100 )
    print("The relative L2 error for t = %f is %f" % ( 3.5*(i+1), relErrorL2 ))




###### NEW ORDER FOR TESTING FD GRADS#########
for i in range(num):
  pointz[i] = np.array( [ np.concatenate( [[xx[j], tt[k]], nuPoints[i][l, 1:]] ) for l in range(len(nuPoints[i])) for k in range(len(tt)) for j in range(len(xx)) ] )

for i in range(num):
  nuPointz[i] = 2 / (UBs - LBs) * (pointz[i] - LBs) - 1

dudmu1stack = np.zeros_like(points)
dudmu2stack = np.zeros_like(points)

for i in range(num):
  u[i] = np.array( [ data[i][l][0][k][j] for l in range(len(points[i])) for k in range(len(tt)) for j in range(len(xx)) ] )

  dudx[i] = np.array( [ data[i][l][4][k][j] for l in range(len(nuPoints[i])) for k in range(len(tt)) for j in range(len(xx)) ] )

  # dudx[i] = np.insert((u[i][1:] - u[i][:-1]) / dx, 0 , 0.)

  # dudx[i] = np.append( np.insert( (u[i][2:] - u[i][:-2]) / (2*dx), 0 , 0. ), (u[i][-1] - u[i][-2]) / dx )

  # def g(x,mu2): return 0.02 * np.exp( mu2 * x )
  # gg[i] = np.array( g(pointz[i][:,0], pointz[i][:,3]) )

  dudmu1stack[i] = np.array( [ bg.compute_dudmu_FD(pointz[i][l][2:], 1e-9)[0] for l in range(len(points[i]))] )
  dudmu2stack[i] = np.array( [ bg.compute_dudmu_FD(pointz[i][l][2:], 1e-9)[1] for l in range(len(points[i]))] )
  dudmu1[i] = np.array( [ dudmu1stack[i][l][k][j] for l in range(len(points[i])) for k in range(len(tt)) for j in range(len(xx)) ] )
  dudmu2[i] = np.array( [ dudmu2stack[i][l][k][j] for l in range(len(points[i])) for k in range(len(tt)) for j in range(len(xx)) ] )
  # dudmu1[i] = np.array( [data[i][l][1][k][j] for j in range(len(xx)) for k in range(len(tt)) for l in range(len(points[i]))] )
  # dudmu2[i] = np.array( [data[i][l][2][k][j] for j in range(len(xx)) for k in range(len(tt)) for l in range(len(points[i]))] )

  dudt[i] = np.array( [data[i][l][3][k][j] for l in range(len(nuPoints[i])) for k in range(len(tt)) for j in range(len(xx)) ] )

  gradu[i] = np.array( [ [dudx[i][j], dudt[i][j], dudmu1[i][j], dudmu2[i][j]] for j in range(len(u[i])) ] )

  nuGradu[i] = 0.5 * (UBs - LBs) * gradu[i]
