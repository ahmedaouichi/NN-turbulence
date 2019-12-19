import numpy as np
from nn import NN
from core import Core
import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

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
    
    eigenvalues_shape = 5
    tensorbasis_shape = 10
    stresstensor_shape = 9

    Retau = 180
    RA_list = [1,3,5,7,10,14]
    velocity_comps = ['U', 'V', 'W']
    DIM = len(velocity_comps)


    print('--> Build network')
    neural_network = NN(8, 15, eigenvalues_shape, tensorbasis_shape, stresstensor_shape)
    neural_network.build()
    
    counter=0
    use_cases = [0,1]
    for RA in  RA_list[0:2]:
        print('Import dataset '+str(counter+1)+'/'+str(len(use_cases)))
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
        
        ## Reshape 2D array to 1D arrays. Network only takes 1D arrays
        stresstensor = np.reshape(stresstensor, (-1, 9))
        tensorbasis = np.reshape(tensorbasis, (-1, 10, 9))
        eigenvalues = np.reshape(eigenvalues, (-1, 5))
        
        ## Input scaling
        scalingfactor = np.max(stresstensor)
        shiftfactor = np.mean(stresstensor)
        stresstensor = stresstensor/scalingfactor
        
        print('--> Train network')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        neural_network.model.fit([tensorbasis, eigenvalues], stresstensor, batch_size = 500, epochs = 20, verbose = 1, shuffle=True, validation_split=0.8, callbacks=[early_stopping])
        
        counter+=1

    ############# Visualization ###############################################

    ## 3rd input argument refers to one of velocity components <U, V, W>
#    core.plotMeanVelocityComponent(RA_list, Retau, 'U')

    # core.plotMeanVelocityField(RA, Retau, data, ycoord, zcoord)

    print('--> Evalutate model')
#    results = neural_network.model.evaluate([tensorbasis, eigenvalues], stresstensor, batch_size=128)
#    print('test loss, test acc:', results)
    
    prediction =  neural_network.model.predict([tensorbasis, eigenvalues])
    prediction = prediction*scalingfactor
    
    
    neural_network.plot_results(prediction, stresstensor)
    
    
    ###############  Visualize prediction  ####################################
    core.tensorplot(stresstensor, DIM_Y, DIM_Z)
    core.tensorplot(prediction, DIM_Y, DIM_Z)
    
    from keras.utils.vis_utils import plot_model
    
    plot_model(neural_network.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    plt.show()
     
main()
