import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from nn import NN
from core import Core
#from keras.models import Sequential
#from keras.layers import Dense
#import keras.backend as K


def main():
    core = Core()
    
    ############# Rectangular Duct data #######################################
    
    ## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'
    
    Retau = 180
    RA_list = [1,3,5,7,10,14]
    velocity_comps = ['U', 'V', 'W']
    DIM = len(velocity_comps)
    
    RA = RA_list[0]
    usecase =  str(RA)+'_'+str(Retau)
    
    print('--> Import grid')
    ycoord, DIM_Y = Core.importCoordinates('y', usecase)
    zcoord, DIM_Z = Core.importCoordinates('z', usecase)
    
    print('--> Import mean velocity field')
    data = np.zeros([DIM_Y, DIM_Z, DIM])
    for ii in range(DIM):
        data[:,:,ii] = core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_comps[ii])
        
    
    print('--> Compute mean velocity gradient')
    velocity_gradient = core.gradient(data, ycoord, zcoord) 
    print('--> Import Reynolds stress tensor')
    tensor = core.importStressTensor(usecase, DIM_Y, DIM_Z)
    
    ############# Visualization ###############################################
    
    ## 3rd input argument refers to one of velocity components <U, V, W>
    #core.plotMeanVelocityComponent(RA_list, Retau, 'U')
    
    #core.plotMeanVelocityField(RA, Retau, data, ycoord, zcoord)
    
    ############ Preparing network inputs #####################################
    
    print('--> Compute omega field')
    eps = np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere
    print('--> Compute k field')
    k = core.calc_k(velocity_gradient)
    print('--> Compute rotation rate and strain rate tensors')
    S,R = core.calc_S_R(velocity_gradient, k, eps)
    print('--> Compute eigenvalues for network input')
    eigenvalues = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
    
    ############  Building neural network  ####################################
    
    #T = core.calc_T(S, R)  # Tensor basis T
   
    #print(input.shape)
    #num_inputs = input.shape[1]
#    num_outputs = b.shape[1]
#    num_layers = 8
#    num_nodes = 30
#    
#    nn = NN(num_layers, num_nodes, num_inputs, num_outputs)
#    
#    #Core.plot_results(b, b)
#    
#    def customLoss(yTrue,yPred):
#        return K.sum((yTrue - yPred)**2)**0.5
#    
#    # load the dataset
#    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
#    # split into input (X) and output (y) variables
#    
#    X = dataset[:,0:8]
#    y = dataset[:,8]
#    # define the keras model
#    model = Sequential()
#    model.add(Dense(12, input_dim=8, activation='relu'))
#    model.add(Dense(8, activation='relu'))
#    model.add(Dense(1, activation='sigmoid'))
#    # compile the keras model
#    model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy'])
#    # fit the keras model on the dataset
#    model.fit(X, y, epochs=10, batch_size=10)
#    # evaluate the keras model
#    _, accuracy = model.evaluate(X, y)
#    print('Accuracy: %.2f' % (accuracy*100))
    
    
main()


