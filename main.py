import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from nn import NN
from core import Core


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
    
    ycoord, DIM_Y = Core.importCoordinates('y', usecase)
    zcoord, DIM_Z = Core.importCoordinates('z', usecase)
    
    data = np.zeros([DIM_Y, DIM_Z, DIM])
        
    for ii in range(DIM):
        data[:,:,ii] = core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_comps[ii])

    ######### Visualization ##########
    
    ## 3rd input argument refers to one of velocity components <U, V, W>
    #core.plotMeanVelocityComponent(RA_list, Retau, 'U')
    
    #core.plotMeanVelocityField(RA, Retau, data, ycoord, zcoord)
    
    grad_u = core.gradient(data, ycoord, zcoord) 
    tensor = core.importStressTensor(usecase, DIM_Y, DIM_Z)
    
    ###########################################################################
    
    ## To be written , for now using dataset
#    k = core.calc_k()
#    eps = core.calc_epsilon()
    
    k, eps, grad_u, stresses = Core.load_test_data()
    
    S,R = core.calc_S_R(grad_u, k, eps)
    
#    eigen_values = core.calc_eigenvalues() ## To be written
#    
#    # nn.set_input(eigen_values)
#    # nn.set_num_hidden_layers(num_layers)
#    # nn.set_num_hidden_nodes(num_nodes)
#    # nn.set_num_in_nodes(num_in_nodes)
#    # nn.set_num_out_nodes(num_out_nodes)
#    # nn.learning_rate(learning_rate)
#    #
#    # output = nn.train()
#    # predict_tau = nn.calc_output(output)
#    
    S, R = core.calc_S_R(grad_u, k, eps)
    input = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
    T = core.calc_T(S, R)  # Tensor basis T
    b = core.calc_output(stresses, k)  # b vector from tau tensor
   
    #print(input.shape)
    num_inputs = input.shape[1]
    num_outputs = b.shape[1]
    num_layers = 8
    num_nodes = 30
    
    nn = NN(num_layers, num_nodes, num_inputs, num_outputs)
    
    #Core.plot_results(b, b)
    
    
main()


