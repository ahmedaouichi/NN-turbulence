import numpy as np
from nn import NN
from processor import Calculator, Plot, Gradient, Read
from core import Core

import matplotlib.pyplot as plt

def importData(usecase):
    velocity_comps = ['U', 'V', 'W']
    
    print('--> Import coordinate system')
    ycoord, DIM_Y = Core.importCoordinates('y', usecase)
    zcoord, DIM_Z = Core.importCoordinates('z', usecase)

    print('--> Import mean velocity field')
    data = np.zeros([DIM_Y, DIM_Z, 3])
    for ii in range(3):
        data[:,:,ii] = Core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_comps[ii])

    print('--> Compute mean velocity gradient')
    velocity_gradient = Gradient.gradient(data, ycoord, zcoord)

    print('--> Import Reynolds stress tensor')
    stresstensor = Core.importStressTensor(usecase, DIM_Y, DIM_Z)
    stresstensor = np.reshape(stresstensor, (-1, 3, 3))

    print('--> Compute omega field')
    eps = 1*np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere

    print('--> Compute k field')
    k = Calculator.calc_k(stresstensor)
    k = np.reshape(k, (DIM_Y, DIM_Z))

    print('--> Compute rotation rate and strain rate tensors')
    S,R = Calculator.calc_S_R(velocity_gradient, k, eps)

    print('--> Compute eigenvalues for network input')
    eigenvalues = Calculator.calc_scalar_basis(S, R)  # Scalar basis lamba's

    print('--> Compute tensor basis')
    tensorbasis = Calculator.calc_tensor_basis(S, R)

    print('--> Compute b')
    k = np.reshape(k, -1)
    b = Calculator.calc_output(stresstensor, k)
    b = np.reshape(b, (-1, 9))

    # for i in range(5):
    #     b = make_realizable(b)

    stresstensor = np.reshape(stresstensor, (-1, 9))
    tensorbasis = np.reshape(tensorbasis, (-1, 10, 9))
    eigenvalues = np.reshape(eigenvalues, (-1, 5))

    return k, b, stresstensor, tensorbasis, eigenvalues, DIM_Y, DIM_Z

def main():
    ## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'
    eigenvalues_shape = 5
    tensorbasis_shape = 10
    stresstensor_shape = 9

    Retau = 180
    RA_list = [1,3,5,7,10,14]
    RA_predict = 5
    RA_list_training = []
    for RA in RA_list:
        if RA != RA_predict:
            RA_list_training.append(RA)

    print('--> Build network')
#    neural_network = NN(8, 30, eigenvalues_shape, tensorbasis_shape, stresstensor_shape)
    
    dim = stresstensor_shape
#    neural_network.build(dim)

    print('--> Train network')

    k, b, stresstensor, tensorbasis, eigenvalues, DIM_Y, DIM_Z = importData(str(1)+'_'+str(180))
    Plot.plotMeanVelocityComponent(RA_list, 180, 'X')
#    for RA in RA_list_training:
#        usecase =  str(RA)+'_'+str(Retau)
#        k, b, stresstensor, tensorbasis, eigenvalues, DIM_Y, DIM_Z = importData(usecase)
#        
#        neural_network.train(eigenvalues, tensorbasis, b)

#    print('--> Plot b prediction')
#    Plot.plot_results(prediction, b)
        
#    usecase =  str(RA_predict)+'_'+str(Retau)
#    k, b, stresstensor, tensorbasis, eigenvalues, DIM_Y, DIM_Z = importData(usecase)
#
#    prediction = neural_network.model.predict([tensorbasis, eigenvalues])
#
#    print('--> Plot stress tensor')
#    Plot.tensorplot(stresstensor, DIM_Y, DIM_Z, 'True stresstensor')
#    
#    predicted_stress = Calculator.calc_tensor(prediction, k)
#    Plot.tensorplot(predicted_stress, DIM_Y, DIM_Z, 'Predicted stresstensor')

    plt.show()
main()
