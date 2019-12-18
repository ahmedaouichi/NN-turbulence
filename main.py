import numpy as np
from nn import NN
from core import Core
import keras
import matplotlib.pyplot as plt

########################## Rectangular Duct data ##############################

def a(x,y):
    return x**2*y**2
    
def b(x,y):
    return x*y

def c(x,y):
    return y**3*x**3

def main():

    
    core = Core()

    ## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'

    Retau = 180
    RA_list = [1,3,5,7,10,14]
    velocity_comps = ['U', 'V', 'W']
    DIM = len(velocity_comps)

    RA = RA_list[0]
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
    print('--> Import Reynolds stress tensor')
    stresstensor = core.importStressTensor(usecase, DIM_Y, DIM_Z)

    ############# Visualization ###############################################

    ## 3rd input argument refers to one of velocity components <U, V, W>
    core.plotMeanVelocityComponent(RA_list, Retau, 'U')

    # core.plotMeanVelocityField(RA, Retau, data, ycoord, zcoord)

    ############ Preparing network inputs #####################################

    print('--> Compute omega field')
    eps = np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere
    print('--> Compute k field')
    k = core.calc_k(velocity_gradient)
    print('--> Compute rotation rate and strain rate tensors')
    S,R = core.calc_S_R(velocity_gradient, k, eps)
    print('--> Compute eigenvalues for network input')
    eigenvalues = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
    print('--> Compute tensor basis')
    tensorbasis = core.calc_tensor_basis(S, R)

    eigenvalues_shape = 5
    tensorbasis_shape = 10
    stresstensor_shape = 9
    
    ## Input scaling
#    scalingfactor = np.max(stresstensor)
#    shiftfactor = np.mean(stresstensor)
#    stresstensor = (stresstensor-shiftfactor)/scalingfactor
#    
    scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
    for i in range(10):
        tensorbasis[:, i, :, :] /= scale_factor[i]
    
    ## Reshape 2D array to 1D arrays. Network only takes 1D arrays
    stresstensor = np.reshape(stresstensor, (-1, 9))
    tensorbasis = np.reshape(tensorbasis, (-1, 10, 9))
    eigenvalues = np.reshape(eigenvalues, (-1, 5))
    

    ############  Building neural network  ####################################
    
    print('--> Build network')
    neural_network = NN(8, 30, eigenvalues_shape, tensorbasis_shape, stresstensor_shape)
    
    print('--> Train network')
    neural_network.build(eigenvalues, tensorbasis, stresstensor)    
    
    print('--> Evalutate model')
    prediction =  neural_network.model.predict([tensorbasis, eigenvalues])
    
    neural_network.plot_results(prediction, stresstensor)
    
    
    ###############  Visualize prediction  ####################################
    core.tensorplot(stresstensor, DIM_Y, DIM_Z)
    core.tensorplot(prediction, DIM_Y, DIM_Z)
    
     
main()
