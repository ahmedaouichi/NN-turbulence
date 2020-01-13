import numpy as np
from nn import NN
from core import Core
import matplotlib.pyplot as plt

def plot_results(predicted_stresses, true_stresses):

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    on_diag = [0, 4, 8]
    for i in range(9):
            plt.subplot(3, 3, i+1)
            ax = fig.gca()
            ax.set_aspect('equal')
            plt.plot([-1., 1.], [-1., 1.], 'r--')
            plt.scatter(true_stresses[:, i], predicted_stresses[:, i])
            plt.xlabel('True value')
            plt.ylabel('Predicted value')
            if i in on_diag:
                plt.xlim([-1./3., 2./3.])
                plt.ylim([-1./3., 2./3.])
            else:
                plt.xlim([-0.5, 0.5])
                plt.ylim([-0.5, 0.5])

#def main():




#main()

core = Core()

## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'
eigenvalues_shape = 5
tensorbasis_shape = 10
stresstensor_shape = 9

Retau = 180
RA_list = [1,3,5,7,10,14]
velocity_comps = ['U', 'V', 'W']
DIM = len(velocity_comps)
predict_index = 3

#RA = RA_list[0]

print('--> Build network')
nodes = 30
layers = 8

eigenvalues_shape = 6
tensorbasis_shape = 10
stresstensor_shape = 9

neural_network = NN(layers, nodes, eigenvalues_shape, tensorbasis_shape, stresstensor_shape)
neural_network.build()

RA_list_loop = []
for i,x in enumerate(RA_list):
    if i!= predict_index:
        RA_list_loop.append(x)

for RA in RA_list_loop:
    usecase =  str(RA)+'_'+str(Retau)
    
    print('--> Import coordinate system')
    ycoord, DIM_Y = Core.importCoordinates('y', usecase)
    zcoord, DIM_Z = Core.importCoordinates('z', usecase)
    
    print('--> Import mean velocity field')
    data = np.zeros([DIM_Y, DIM_Z, DIM])
    for ii in range(DIM):
        data[:,:,ii] = core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_comps[ii])
    
    print('--> Compute mean velocity gradient')
    velocity_gradient = core.gradient(data, ycoord, zcoord)
    
    core.tensorplot(velocity_gradient, DIM_Y, DIM_Z, 'Velocity gradient')
    
    print('--> Import Reynolds stress tensor')
    stresstensor = core.importStressTensor(usecase, DIM_Y, DIM_Z)
    stresstensor = np.reshape(stresstensor, (-1, 3, 3))
    
    ############ Preparing network inputs #####################################
    
    print('--> Compute omega field')
    eps = 1*np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere
    print('--> Compute k field')
    # k = core.calc_k(stresstensor) ## In realitiy this is not possible, because k is unknown
    k = 1*np.ones([DIM_Y, DIM_Z]) ## For now using k=1 everywhere    
    k = np.reshape(k, (DIM_Y, DIM_Z))
    print('--> Compute rotation rate and strain rate tensors')
    S,R = core.calc_S_R(velocity_gradient, k, eps)
    print('--> Compute eigenvalues for network input')
    eigenvalues = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
    print('--> Compute tensor basis')
    tensorbasis = core.calc_tensor_basis(S, R)
    
    # Plot tensorbasis
    #for x in range(0,10):
    #    core.tensorplot(tensorbasis[:,:,x], DIM_Y, DIM_Z, str(x))
    
    core.tensorplot(S[:,:,:,:], DIM_Y, DIM_Z, 'Strain rate tensor')
    core.tensorplot(R[:,:,:,:], DIM_Y, DIM_Z, 'Rotation rate tensor')
    
    ## Reshape 2D array to 1D arrays. Network only takes 1D arrays
    k = np.reshape(k, -1)
    #b = core.calc_output(stresstensor, k)
    stresstensor = np.reshape(stresstensor, (-1, 9))
    tensorbasis = np.reshape(tensorbasis, (-1, 10, 9))
    eigenvalues = np.reshape(eigenvalues, (-1, 6))
    
    print('--> Train network')
    neural_network.train(eigenvalues, tensorbasis, stresstensor)

RA = RA_list[predict_index]
usecase =  str(RA)+'_'+str(Retau)
    
print('--> Import coordinate system')
ycoord, DIM_Y = Core.importCoordinates('y', usecase)
zcoord, DIM_Z = Core.importCoordinates('z', usecase)

print('--> Import mean velocity field')
data = np.zeros([DIM_Y, DIM_Z, DIM])
for ii in range(DIM):
    data[:,:,ii] = core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_comps[ii])

print('--> Compute mean velocity gradient')
velocity_gradient = core.gradient(data, ycoord, zcoord)

core.tensorplot(velocity_gradient, DIM_Y, DIM_Z, 'Velocity gradient')

############ Preparing network inputs #####################################

print('--> Compute omega field')
eps = 1*np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere
print('--> Compute k field')
# k = core.calc_k(stresstensor) ## In realitiy this is not possible, because k is unknown
k = 1*np.ones([DIM_Y, DIM_Z]) ## For now using k=1 everywhere    
k = np.reshape(k, (DIM_Y, DIM_Z))
print('--> Compute rotation rate and strain rate tensors')
S,R = core.calc_S_R(velocity_gradient, k, eps)
print('--> Compute eigenvalues for network input')
eigenvalues = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
print('--> Compute tensor basis')
tensorbasis = core.calc_tensor_basis(S, R)

# Plot tensorbasis
#for x in range(0,10):
#    core.tensorplot(tensorbasis[:,:,x], DIM_Y, DIM_Z, str(x))

#core.tensorplot(S[:,:,:,:], DIM_Y, DIM_Z, 'Strain rate tensor')
#core.tensorplot(R[:,:,:,:], DIM_Y, DIM_Z, 'Rotation rate tensor')

## Reshape 2D array to 1D arrays. Network only takes 1D arrays
k = np.reshape(k, -1)
tensorbasis = np.reshape(tensorbasis, (-1, 10, 9))
eigenvalues = np.reshape(eigenvalues, (-1, 6))


prediction = neural_network.model.predict([tensorbasis, eigenvalues])

#core.tensorplot(stresstensor, DIM_Y, DIM_Z, 'True stresstensor')
predicted_stress = core.calc_tensor(prediction, k)
core.tensorplot(prediction, DIM_Y, DIM_Z, 'Predicted stresstensor')

###############  Visualize prediction  ####################################

from keras.utils.vis_utils import plot_model

plot_model(neural_network.model, to_file='model.png', show_shapes=True, show_layer_names=True)

plt.show()

