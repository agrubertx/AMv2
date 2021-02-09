import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


### EXAMPLE 1, PYTORCH AUTOENCODER
### https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 512
epochs = 20
learning_rate = 1e-3

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=784).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

optimizer

# mean-squared error loss
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=False
)

test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, 784)
        reconstruction = model(test_examples)
        break


with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



### EXAMPLE 2, PYTORCH 60 MIN TUTORIAL
### https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    # 1 input image channel, 6 output channels, 3x3 square convolution
    # kernel
    self.conv1 = nn.Conv2d(1, 6, 3)
    self.conv2 = nn.Conv2d(6, 16, 3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    # Max pooling over a (2, 2) window
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    # If the size is a square you can only specify a single number
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
net.zero_grad()
out.backward(torch.randn(1, 10))

print(out)

data = [[[1, 2, 3],[1, 3, 4]],[[2, 2, 3],[2, 3, 4]]]
x_data = torch.tensor(data)
x_data.shape
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones}")

import torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
data.shape

prediction = model(data) # forward pass
len(prediction[0])

loss = ((prediction - labels)**2).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
  tensor = tensor.to('cuda')

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
tensor.matmul(tensor.T)
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

import numpy as np
import numpy.matlib as npm
import numethod as nm
import basicFuncs as bf
import pandas as pd
import scipy as sy
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import active_subspaces as ac
import importlib
importlib.reload(ac)
import torch

import os
print(os.path.abspath("."))

nm.get_random_init_pt(12,34)

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

importlib.reload(ac)
meshh = nm.make_mesh(2, 10)


nm.sample_f_on_mesh(bf.Squaresum, meshh)

reload(nm)
mesh, fSamples, paths, realgrads = nm.build_random_data( 3, 100, bf.Squaresum, bf.gradSquaresum)
nm.subspEx( mesh, fSamples, realgrads, 0.1, 5, nm.get_random_init_pt(3))

mesh, fSamples, paths, realgrads = nm.build_random_data( 2, 1000, bf.uhoh, bf.graduhoh)

X_train, X_test, y_train, y_test, meshgrads, testgrads = model_selection.train_test_split(
    mesh, fSamples, realgrads, test_size = 0.2, random_state = 0)

meshgrads

# Compute gradient
gradPaths = preprocessing.normalize( meshgrads)

mesh_kdtree = sy.spatial.KDTree( X_train)

def get_f_value(startPoint, mesh_kdtree, fSamples, realGrads):
    cdist,i = mesh_kdtree.query(startPoint)
    c = mesh_kdtree.data[i]
    fStartPoint = fSamples[i] + np.dot(bf.gradSquaresum(*(startPoint+c)/2), startPoint - c)
    #fStartPoint = fSamples[i] + np.dot(bf.gradSquaresum(*(startPoint+c)), startPoint - c)
    return fStartPoint

fApproxVals = [get_f_value(tp, mesh_kdtree, y_train, meshgrads) for tp in X_test]

    # Average L1, L2 error of the fit
    fitErrorL1 = np.mean(np.abs(y_test - fApproxVals))
    fitErrorL2 = np.linalg.norm(fApproxVals - y_test) / float(len(X_test))

    # Relative L1, L2 errors
    relErrorL1 = np.mean(np.abs(y_test - fApproxVals)) / np.mean(np.abs(y_test))
    relErrorL2 = np.linalg.norm(fApproxVals - y_test) / np.linalg.norm(y_test)

relErrorL1
relErrorL2
    # # Build active manifold
    # activeManifold, fCloseVals = build_AM_from_data(
    #     seedPoint=seedPoint, mesh_kdtree=mesh_kdtree,
    #     fSamples=fSampsTraining, gradPaths=gradPaths, realGrads=meshgrads, stepsize=stepsize)
    #
    # # Fit to model function
    # numpts = len(activeManifold)
    # sValues = np.linspace(0., numpts, numpts) / (numpts)
    # splinef = sy.interpolate.PchipInterpolator(sValues, fCloseVals)

fApproxVals = [ nm.get_f_value( tp, mesh_kdtree, fSampsTraining, meshgrads) for tp in testPoints]

# Average L1, L2 error of the fit
fitErrorL1 = np.mean( np.abs( fSampsTest - fApproxVals))
fitErrorL2 = np.linalg.norm( fApproxVals - fSampsTest) / float( len( testPoints))

fitErrorL1
fitErrorL2

bf.Squaresum(*testPoints[250])
fSampsTest[250]
len(fApproxVals)

am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(12), mesh, fSamples, paths, realgrads, 0.1)
nm.splinePlot( am, fvals, 'squaresum', '{ss}')
nm.subspEx( mesh, fSamples, realgrads, 1, 0, .2)

mesh_kdtree = sy.spatial.KDTree(mesh)

    # Define region / hypercube [-1,1]^(m+1)
    dim = len( nm.get_random_init_pt(12))
    rBound = np.ones(dim)
    #rBound = np.append(np.ones(liftDim - 1),t_0)
    Dom = sy.spatial.Rectangle( -1*rBound, rBound )
    # Define starting point
    p0 =  nm.get_random_init_pt(12)
    # Find closest mesh point to seedpoint (i0 is index)
    # Use d0 for first direction
    i0 = mesh_kdtree.query(p0)[1]
    c0 = mesh_kdtree.data[i0]
    d0 = paths[i0]

    # Line integral to get f(p0).
    fp0 = fSamples[i0] + np.dot(realgrads[i0], p0-c0)
    mid = (p0+c0)/2
fpp0 = fSamples[i0] + np.dot(bf.gradSquaresum(*c0), mid - c0) + np.dot(bf.gradSquaresum(*mid), p0 - mid)
fp0
fpp0
bf.Squaresum(*p0)
fSamples[i0]
np.array(1)

    # Initialize gradient ascent
    ascentPoints = np.asarray(p0)
    aCloseVals = np.asarray(fp0)

    # Take one step
    p1 = p0 + (0.1 * d0)

    i1 = mesh_kdtree.query(p1)[1]
    c1 = mesh_kdtree.data[i1]

    fp1 = fSamples[i1] + np.dot(realgrads[i1], p1-c1)

fp1
fSamples[i1]
bf.Squaresum(*p1)
    ascentPoints = np.vstack((ascentPoints,p1))
    aCloseVals = np.append(aCloseVals, fp1)

ascentPoints
aCloseVals

i0
iOld
i1
    cond = np.array(1)
    # Gradient ascent, computing line integrals to recover f along the AM
    while Dom.min_distance_point(ascentPoints[-1]) == 0 and min(cond.flatten()) > 0.05/2:

        iOld = mesh_kdtree.query(ascentPoints[-1])[1]
        d = paths[iOld]

        p = ascentPoints[-1] + (0.05 * d)

        i = mesh_kdtree.query(p)[1]
        c = mesh_kdtree.data[i]

        fp = fSamples[i] + np.dot(realgrads[i], p - c)

        ascentPoints = np.vstack((ascentPoints, p))
        aCloseVals = np.append(aCloseVals, fp)

        #update loop condition
        cond = sy.spatial.distance.cdist([ascentPoints[-1]], ascentPoints[0:len(ascentPoints)-1], 'euclidean')

ascentPoints

# Delete last elements (outside of hypercube)
ascentPoints = np.delete(ascentPoints, len(ascentPoints) - 1, 0)
aCloseVals = np.delete(aCloseVals, len(aCloseVals) - 1, 0)

    # Initialize gradient descent
    descentPoints = np.asarray(p0)
    dCloseVals = fp0

    # Take one step
    p1 = p0 - 0.05 * d0

    i1 = mesh_kdtree.query(p1)[1]
    c1 = mesh_kdtree.data[i1]

    fp1 = fSamples[i1] + np.dot(realgrads[i1], p1-c1)

    descentPoints = np.vstack((descentPoints,p1))
    dCloseVals = np.append(dCloseVals, fp1)

dCloseVals
descentPoints
i1
iOld
i
    # Gradient descent, using line integrals to recover f along the AM
    while Dom.min_distance_point(descentPoints[-1]) == 0 and min(cond.flatten()) > 0.05/2:

        iOld = mesh_kdtree.query(descentPoints[-1])[1]
        d = paths[iOld]

        p = descentPoints[-1] - (0.05 * d)

        i = mesh_kdtree.query(p)[1]
        c = mesh_kdtree.data[i]

        fp = fSamples[i] + np.dot(realgrads[i], p - c)

        descentPoints = np.vstack((descentPoints,p))
        dCloseVals = np.append(dCloseVals, fp)

        #update loop condition
        cond = sy.spatial.distance.cdist([descentPoints[-1]], descentPoints[0:len(descentPoints)-1], 'euclidean')

    # Delete first and last element in descentpoints and fValuesdescent
    descentPoints = np.delete(descentPoints, 0, 0)
    descentPoints = np.delete(descentPoints, len(descentPoints) - 1, 0)
    dCloseVals = np.delete(dCloseVals, 0)
    dCloseVals = np.delete(dCloseVals, len(dCloseVals) - 1)

dCloseVals

i
fSamples[1256]
c
realgrads[1256]
p-c
np.dot(realgrads[i], p - c)
descentPoints[3]
dCloseVals

    # Flip order of descentPoints and concatenate lists
    activeManifold = np.concatenate((np.flipud(descentPoints), ascentPoints), axis=0)
    fCloseVals = np.concatenate((np.flipud(dCloseVals), aCloseVals))
bf.Squaresum(*activeManifold)[1]
min([bf.Squaresum(*i) for i in activeManifold])
max(bf.Squaresum(*activeManifold))
fCloseVals[1]
dCloseVals[10]
    return activeManifold, fCloseVals
fCloseVals[9]
fCloseVals[10]
bf.Squaresum(*activeManifold[9])
nm.splinePlot( activeManifold, fCloseVals, 'squaresum', '{ss}')

len(y_test)

    ss = ac.subspaces.Subspaces()
    ss.compute( df=meshgrads, nboot=100 )
    ss.partition(1)
    xRed = np.dot( X_train, ss.W1 )

    # Build polynomial approximation to the data, using Constantine functions
    RS = ac.subspaces.PolynomialApproximation(4)
    #RS = ac.utils.response_surfaces.RadialBasisApproximation(2)
    y_train = y_train.reshape( ( len(y_train), 1 ) )
    RS.train( xRed, y_train )
    print 'The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr)

    # Plot it
    if dim == 1:
        plt.figure( figsize=(7, 7) )
        y0 = np.linspace( -1, 1, 200 )
        plt.scatter( xRed, y_train, c = '#66c2a5' )
        plt.plot( y0, RS.predict(y0[:, None])[0], c = '#fc8d62', ls='-', linewidth=2 )
        plt.grid(True)
        plt.xlabel( 'Active Variable Value', fontsize=18 )
        plt.ylabel( 'Output', fontsize=18 )

    # Build active subspace from trainingPoints with 100 bootstrap replicates
    ss = ac.subspaces.Subspaces()
    ss.compute(df=meshgrads, sstype='AS')
ac.utils.plotters.eigenvalues(ss.eigenvals)
    ss.partition(2)
    y = np.dot(mesh,ss.W1)
len(ss.W1[0])

# set up the active variable domain
avd = ac.domains.BoundedActiveVariableDomain(ss)
# set up the maps between active and full variables
avm = ac.domains.BoundedActiveVariableMap(avd)

rs = ac.response_surfaces.ActiveSubspaceResponseSurface(avm)
# or train with the existing runs
rs.train_with_data(X_train, y_train.reshape((len(y_train),1)))

    fApproxVals = rs.predict(X_test, False)[0]

fApproxVals[4]
y_test[4]
    # Average L1, L2 errors
    fitErrorL1 = np.mean(np.abs(y_test - fApproxVals))
    fitErrorL2 = np.linalg.norm(fApproxVals - y_test) / float(len(X_test))

fTestVals = y_test.reshape( (len(y_test), 1)

np.linalg.norm(y_test)
np.linalg.norm(fApproxVals - y_test)
    # Relative L1, L2 errors
    relErrorL1 = np.sum(np.abs(y_test.reshape( (len(y_test), 1))) - fApproxVals) / np.sum(np.abs(y_test))
    relErrorL2 = np.linalg.norm(np.abs(y_test.reshape( (len(y_test), 1)) - fApproxVals)) / np.linalg.norm(y_test)
    relErrorL1t = np.mean(np.abs(y_test - fApproxVals)) / np.mean(np.abs(fApproxVals))
    relErrorL2t = np.linalg.norm((fApproxVals - y_test)/len(X_test)) / np.linalg.norm(fApproxVals) len(X_test))

relErrorL1
relErrorL2
relErrorL2t
ac.utils.plotters.sufficient_summary(y, fSamples)
len(ss.W1)
ss.train_with_data(mesh, fSamples)
# set up the active variable domain
avd = ac.domains.BoundedActiveVariableDomain(ss)
# set up the maps between active and full variables
avm = ac.domains.BoundedActiveVariableMap(avd)
    # Build polynomial approximation to the data, using Constantine functions
    RS = ac.subspaces.PolynomialApproximation(4)
    y_train = y_train.reshape((len(y_train),1))
    RS.train(y, y_train)
    print 'The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr)


#### 20 dim ###

def sinfunc(*vars):
	total = 0
	for i in range(0,len(vars)):
		total += (vars[i])**2
	return np.sin(total)

def cosfunc(*vars):
	total = 0
	for i in range(0,len(vars)):
		total += (vars[i])**2
	return np.cos(total)

def gradsinfunc(*vars):
	grs = []
	for i in xrange(len(vars)):
		grs =np.concatenate( (grs, np.array([2*vars[i] * cosfunc(*vars)])), axis = 0)
	return grs

mesh, fSamples, paths, realgrads = nm.build_random_data( 20, 1000, sinfunc, gradsinfunc)
am,fvals = nm.build_AM_from_data( nm.get_random_init_pt(20), mesh, fSamples, paths, realgrads, 0.01)
nm.splinePlot( am, fvals, 'sinfunc', '{sf}')
nm.mainRandEx_old( mesh, fSamples, realgrads, 0.1, 0.2, nm.get_random_init_pt(20))
nm.subspEx( mesh, fSamples, realgrads, 1, 0, .2)




        f1 = 2.0*np.exp(-1 * np.sum((mesh-0.0) * (mesh-0.0), axis=1) * 2.0)
        f2 = 2.0*np.exp(-1 * np.sum((mesh-1.0) * (mesh-1.0), axis=1) * 2.0)
        f = f1 + f2
        df = -8.0 * (mesh-0.0) * npm.repmat(np.expand_dims(f1, axis=1),1,20) -8.0 * (mesh-1.0) * npm.repmat(np.expand_dims(f2, axis=1),1,20)

cc = 0.1
ww = 0.0

f = np.cos(np.sum(mesh*cc,axis=1))
df = npm.repmat(np.expand_dims(-1 * cc * np.sin(np.sum( mesh, axis=1)), axis=1),1,20)

cc = 1.2
ww = 0.0
f = np.prod( (cc**(-2.0) + (mesh-ww)**2.0)**(-1.0), axis=1)
df = npm.repmat(np.expand_dims(f,axis=1),1,20) * -1.0 * (cc**-2.0 + (mesh-ww)**2.0)**-1.0 * 2.0 * (mesh-ww)

mesh, fSamples, paths, realgrads = nm.build_random_data( 2, 6, sinfunc, gradsinfunc)
len(mesh)

nm.mainRandEx_old( mesh, f, df, 0.1, 0.2, nm.get_random_init_pt(20))

    # Build active subspace from trainingPoints
    ss = ac.subspaces.Subspaces()
    ss.compute(df=df, sstype='AS')

    #Plot Eigenvals
    ac.utils.plotters.eigenvalues(ss.eigenvals)
    dimension = len(ss.W1[0])
    print 'The dimension of the Active Subspace is %i' %dimension

    # Plot it
    if dimension < 3:
        domVals = np.dot(mesh, ss.W1)
        ac.utils.plotters.sufficient_summary(domVals, f)

    # set up the active variable domain
    avd = ac.domains.BoundedActiveVariableDomain(ss)
    # set up the maps between active and full variables
    avm = ac.domains.BoundedActiveVariableMap(avd)

    #Build and train response surface
    rs = ac.response_surfaces.ActiveSubspaceResponseSurface(avm)
    rs.train_with_data(mesh, f.reshape((len(f),1)))

    #Compute approximate fVals
    X_test = np.random.uniform(0., 1.0, (10000, 20) )
    y_test = sinfunc(*X_test)
    fApproxVals = rs.predict(X_test)[0]

    # Average L1, L2 errors
    fitErrorL1 = np.mean(np.abs(y_test - fApproxVals))
    fitErrorL2 = np.linalg.norm(fApproxVals - y_test) / float(len(X_test))

    # Relative L1, L2 errors
    relErrorL1 = np.mean(np.abs(y_test - fApproxVals)) / np.mean(np.abs(y_test))
    relErrorL2 = np.linalg.norm(fApproxVals - y_test) / np.linalg.norm(y_test)

    results = {'fit L1 error': fitErrorL1, "fit L2 error": fitErrorL2}
    if outpath:
        jsonify(results, outpath)
    #
    print 'The Relative L1 Error is %f' %relErrorL1
    print 'The Relative L2 Error is %f' %relErrorL2
    print 'The L1 Error is %f' %fitErrorL1
    print 'The L2 Error is %f' %fitErrorL2

x = mesh

        f = x[:,0]**3 + x[:,1]**3 + x[:,0] * 0.2 + 0.6 * x[:,1]
        df = npm.repmat(np.expand_dims(f,axis=1),1,2)
f
np.expand_dims(f,axis=1)
df
        df[:,0] = 3.0*x[:,0]**2.0 + 0.2
        df[:,1] = 3.0*x[:,1]**2.0 + 0.6
df
# x = np.linspace(0, 6*np.pi, 100)



import burgers as bg


h = 0.0000001

UU, DDsav, QQ, gradQQ = burgers(2, 256, 2000, 35, 1, [5.5, 0.021], False)
QQf = burgers(2, 256, 2000, 35, 1, [5.5, 0.021+h], False)[2]
QQb = burgers(2, 256, 2000, 35, 1, [5.5, 0.021-h], False)[2]
test = (QQf - QQb) / (2*h)
test
gradQQ
abs(test - gradQQ[2]) / test
burgers(2, 512, 2000, 35, 1, [5.5, 0.021], False)[2:]

sample = np.random.uniform(-1, 1, (200, 3))
UBs = np.array([45, 5.5, 0.03])
LBs = np.array([25, 4.25, 0.015])

nuSample = LBs + 0.5 * (UBs - LBs) * (1 + sample)

cool2 = map(lambda x: bg.generate_fd_gradients(h, nuSample[x,0], nuSample[x,1:]), range(len(nuSample)))
testQ

cool = map(lambda x: bg.burgers(2, 256, 2000, nuSample[x,0], 1, nuSample[x,1:], False)[2:], range(len(nuSample)))

sampleQ = np.array([cool2[i][0] for i in range(len(cool))])
realGrads = [cool2[i][1] for i in range(len(cool))]

nuGrads = 0.5 * (UBs - LBs) * realGrads


nm.mainRandEx_old( sample, sampleQ, nuGrads, 0.1, 50, nm.get_random_init_pt(3))
am, fvals = nm.build_AM_from_data( nm.get_random_init_pt(3), sample, sampleQ, preprocessing.normalize(nuGrads), nuGrads, 0.005)
nm.splinePlot( am, fvals,'burgers','{b}')
nm.subspEx(sample, np.array(sampleQ), nuGrads, 2, 0, .25)
nuuSample = nuSample[:25]
ssample = sample[:25]

ccool = map(lambda x: bg.burgers(2, 256, 2000, nuuSample[x,0], 1, nuuSample[x,1:], False)[2:], range(len(nuuSample)))

sampleQ = np.array([ccool[i][0] for i in range(len(ccool))])
realGrads = [ccool[i][1] for i in range(len(ccool))]

nuGrads = 0.5 * (UBs - LBs) * realGrads


nm.mainRandEx_old( ssample, sampleQ, nuGrads, 0.02, 5, nm.get_random_init_pt(3) )
nm.subspEx(ssample, sampleQ, nuGrads, 2, 0, .2)

def mainRandEx_old( inputs, fSamples, grads, stepsize, test_size, seedPoint,
    seed = 0, plot = True, outpath = None, verbose = False ):
    """
    Anthony's function for testing error between function and active manifold
    """

    # Single-step line integral to recover function value at unknown points
    def get_f_value( startPoint, mesh_kdtree, fSamples, realGrads ):
        cdist,i = mesh_kdtree.query(startPoint)
        c = mesh_kdtree.data[i]
        fStartPoint = fSamples[i] + np.dot( realGrads[i], startPoint - c )

        return fStartPoint

    # Random Samples to Test function approximation on
    meshPoints, testPoints, fSampsTraining, fSampsTest, meshGrads, testGrads = model_selection.train_test_split(
        mesh, fSamples, realgrads, test_size = 0.2, random_state = 0 )

    # Compute normalized gradient
    gradPaths = preprocessing.normalize(meshGrads)

    mesh_kdtree = sy.spatial.KDTree(meshPoints)

    # Build active manifold
    if plot:
        activeManifold, fValsAM = build_AM_from_data(
            seedPoint=seedPoint, meshPoints=meshPoints,
            fSamples=fSampsTraining, realGrads=meshGrads, stepsize=stepsize)

        splinePlot( activeManifold, fValsAM, "function", "{}" )

        # # Fit to model function
        # numpts = len(activeManifold)
        # sValues = np.linspace(0., numpts, numpts) / (numpts)
        # splinef = sy.interpolate.PchipInterpolator(sValues, fCloseVals)

    fApproxVals = [ get_f_value( tp, mesh_kdtree, fSampsTraining, meshGrads ) for tp in testPoints ]

fApproxVals - fSampsTest

    # Average L1, L2 error of the fit
    fitErrorL1 = np.mean( np.abs( fSampsTest - fApproxVals ) )
    fitErrorL2 = np.linalg.norm( fApproxVals - fSampsTest ) / float( len(testPoints) )

    # Relative L1, L2 errors
    # relErrorL1 = np.mean( np.abs( fSampsTest - fApproxVals) ) / np.mean( np.abs(fSampsTest) ) * 100
    relErrorL1 = np.sum( np.abs( fSampsTest - fApproxVals) ) / np.sum( np.abs(fSampsTest) ) * 100
    relErrorL2 = np.linalg.norm( fApproxVals - fSampsTest ) / np.linalg.norm(fSampsTest) * 100

# Test for fixed-time energy density -- Q3

sample200 = np.random.uniform( -1, 1, (200, 3) )
sample100 = np.random.uniform( -1, 1, (100, 3) )
sample50 = np.random.uniform( -1, 1, (50, 3) )
sample25 = np.random.uniform( -1, 1, (25, 3) )
sample10 = np.random.uniform( -1, 1, (10, 3) )

# Upper and Lower bounds for parameters
UBs = np.array( [35, 5.5, 0.03] )
LBs = np.array( [0, 4.25, 0.015] )

# Linear scaling to appropriate range
nuSample200 = LBs + 0.5 * (UBs - LBs) * (1 + sample200)
nuSample100 = LBs + 0.5 * (UBs - LBs) * (1 + sample100)
nuSample50 = LBs + 0.5 * (UBs - LBs) * (1 + sample50)
nuSample25 = LBs + 0.5 * (UBs - LBs) * (1 + sample25)
nuSample10 = LBs + 0.5 * (UBs - LBs) * (1 + sample10)

dt = 35/2000

data200 = map( lambda x: bg.burgers( 2, 256, 2000, 35, 1, nuSample200[x, 1:],
    False)[2:], range( len(nuSample200) ) )
data100 = map( lambda x: bg.burgers( 2, 256, 2000, 35, 1, nuSample100[x, 1:],
    False)[2:], range( len(nuSample100) ) )
data50 = map( lambda x: bg.burgers( 2, 256, 2000, 35, 1, nuSample50[x, 1:],
    False)[2:], range( len(nuSample50) ) )
data25 = map( lambda x: bg.burgers( 2, 256, 2000, 35, 1, nuSample25[x, 1:],
    False)[2:], range( len(nuSample25) ) )
data10 = map( lambda x: bg.burgers( 2, 256, 2000, 35, 1, nuSample10[x, 1:],
    False)[2:], range( len(nuSample10) ) )

# data [simulation j][quantity returned][time step i] (recall node is fixed at rightend)
# quantity 5 has an additional index for component of the gradient
# returns [Q3, dQ3dt, dQ3dmu1, dQ3dmu2] at 2 random times from each simulation
reduceData200 = np.array( [ [dt*i, data200[j][4][i], data200[j][5][0][i], data200[j][5][1][i],
  data200[j][5][2][i]] for i in np.random.choice(range(len(data10[0][5][0])), 2) for j in range(len(data200)) ] )
reduceData100 = np.array( [ [dt*i, data100[j][4][i], data100[j][5][0][i], data100[j][5][1][i],
  data100[j][5][2][i]] for i in np.random.choice(range(len(data100[0][5][0])), 2) for j in range(len(data100)) ] )
reduceData50 = np.array( [ [dt*i, data50[j][4][i], data50[j][5][0][i], data50[j][5][1][i],
  data50[j][5][2][i]] for i in np.random.choice(range(len(data50[0][5][0])), 2) for j in
  range(len(data50)) ] )
reduceData25 = np.array( [ [dt*i, data25[j][4][i], data25[j][5][0][i], data25[j][5][1][i],
  data25[j][5][2][i]] for i in np.random.choice(range(len(data25[0][5][0])), 2) for j in
  range(len(data25)) ] )
reduceData10 = np.array( [ [i, data10[j][4][i], data10[j][5][0][i], data10[j][5][1][i],
  data10[j][5][2][i]] for i in np.random.choice(range(len(data10[j][5][0])), 2) for j in
  range(len(data10)) ] )

Qt200 = reduceData200[:, 1]
Qt100 = reduceData100[:, 1]
Qt50 = reduceData50[:, 1]
Qt25 = reduceData25[:, 1]
Qt10 = reduceData10[:, 1]
reduceData10[:,0]
gradQt200 = reduceData200[:, 2:]
gradQt100 = reduceData100[:, 2:]
gradQt50 = reduceData50[:, 2:]
gradQt25 = reduceData25[:, 2:]
gradQt10 = reduceData10[:, 2:]

nuGradsT200 = 0.5 * (UBs - LBs) * gradQt200
nuGradsT100 = 0.5 * (UBs - LBs) * gradQt100
nuGradsT50 = 0.5 * (UBs - LBs) * gradQt50
nuGradsT25 = 0.5 * (UBs - LBs) * gradQt25
nuGradsT10 = 0.5 * (UBs - LBs) * gradQt10
reduceData10[6]
np.array( [ reduceData10[0], nuSample10[0][1:] ])


nm.mainRandEx_old( sample200, Qt200, nuGradsT200, 0.1, 0.2, nm.get_random_init_pt(3, 0), 0, False )
nm.subspEx( sample200, Qt200, nuGradsT200, 2, 0, 0.2, False )

nm.mainRandEx_old( sample100, Qt100, nuGradsT100, 0.1, 0.2, nm.get_random_init_pt(3, 0), 0, False )
nm.subspEx( sample100, Qt100, nuGradsT100, 2, 0, 0.2, False )

nm.mainRandEx_old( sample50, Qt50, nuGradsT50, 0.1, 0.2, nm.get_random_init_pt(3, 0), 0, False )
nm.subspEx( sample50, Qt50, nuGradsT50, 2, 0, 0.2, False )

nm.mainRandEx_old( sample25, Qt25, nuGradsT25, 0.1, 0.2, nm.get_random_init_pt(3, 0), 0, False )
nm.subspEx( sample25, Qt25, nuGradsT25, 2, 0, 0.2, False )

nm.mainRandEx_old( sample10, Qt10, nuGradsT10, 0.1, 0.2, nm.get_random_init_pt(3, 0), 0, False )
nm.subspEx( sample10, Qt10, nuGradsT10, 2, 0, 0.2, False )

nm.mainRandEx_old( sample100, Qt100, nuGradsT100, 0.1, 0.2, nm.get_random_init_pt(3, 0), 0 )
nm.subspEx( sample100, Qt100, nuGradsT100, 2, 0, 0.2 )
