import numpy as np
from nn import NN
from core import Core
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.backend as K
import math as m
import matplotlib.pyplot as plt

def main():
    core = Core()

    ## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'
    eigenvalues_shape = 5
    tensorbasis_shape = 10
    stresstensor_shape = 9

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
    stresstensor = np.reshape(stresstensor, (-1, 3, 3))

    print('--> Compute omega field')
    eps = 1*np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere

    print('--> Compute k field')
    k = core.calc_k(stresstensor)
    k = np.reshape(k, (DIM_Y, DIM_Z))

    print('--> Compute rotation rate and strain rate tensors')
    S,R = core.calc_S_R(velocity_gradient, k, eps)

    print('--> Compute eigenvalues for network input')
    eigenvalues = core.calc_scalar_basis(S, R)  # Scalar basis lamba's

    print('--> Compute tensor basis')
    tensorbasis = core.calc_tensor_basis(S, R)

    print('--> Compute b')
    k = np.reshape(k, -1)
    b = core.calc_output(stresstensor, k)
    b = np.reshape(b, (-1, 9))

    # for i in range(5):
    #     b = make_realizable(b)

    stresstensor = np.reshape(stresstensor, (-1, 9))
    tensorbasis = np.reshape(tensorbasis, (-1, 10, 9))
    eigenvalues = np.reshape(eigenvalues, (-1, 5))

    print('--> Build network')
    neural_network = NN(2, 20, eigenvalues.shape[1], tensorbasis.shape[1], b.shape[1])
    dim = b.shape[0]
    neural_network.build(dim)

    print('--> Train network')
    # for i in range(5)
    prediction = neural_network.train(eigenvalues, tensorbasis, b)

    # for i in range(5):
    #     prediction = make_realizable(prediction)

    print('--> Plot b prediction')
#    core.plot_results(prediction, b)
    
    print('--> Plot b tensor')
#    core.tensorplot(b, DIM_Y, DIM_Z, 'True anisotropy tensor')
#    core.tensorplot(prediction, DIM_Y, DIM_Z, 'Predicted anisotropy tensor')

    print('--> Plot stress tensor')
    core.tensorplot(stresstensor, DIM_Y, DIM_Z, 'True stresstensor')
    predicted_stress = core.calc_tensor(prediction, k)
    core.tensorplot(predicted_stress, DIM_Y, DIM_Z, 'Predicted stresstensor')

    plt.show()
main()
