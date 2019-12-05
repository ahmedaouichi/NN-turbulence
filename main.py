import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from nn import NN
from core import Core


def load_test_data():
    data = np.loadtxt("nn/test_data.txt", skiprows=4)
    k = data[:, 0]
    eps = data[:, 1]
    grad_u_flat = data[:, 2:11]
    stresses_flat = data[:, 11:]

    num_points = data.shape[0]
    grad_u = np.zeros((num_points, 3, 3))
    stresses = np.zeros((num_points, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_u[:, i, j] = grad_u_flat[:, i*3+j]
            stresses[:, i, j] = stresses_flat[:, i*3+j]

    return k, eps, grad_u, stresses

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
            idx_1 = i / 3
            idx_2 = i % 3
            plt.title('A' + str(idx_1) + str(idx_2))
            if i in on_diag:
                plt.xlim([-1./3., 2./3.])
                plt.ylim([-1./3., 2./3.])
            else:
                plt.xlim([-0.5, 0.5])
                plt.ylim([-0.5, 0.5])
    plt.tight_layout()
    plt.show()


def main():

    k, eps, grad_u, stresses = load_test_data()

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
    core.plotMeanVelocityComponent(RA_list, Retau, 'U')
    
    core.plotMeanVelocityField(RA, Retau, data, ycoord, zcoord)
    
    grad_u = core.gradient(data, ycoord, zcoord) 
    
    ###########################################################################
    
    # tau = core.get_tau()
    # u = core.get_u()
    
    # k = core.calc_k()
    # eps = core.calc_epsilon()
    # n = core.get_n()
    # S,R = core.calc_S_R(grad_u, k, eps, n)
    # eigen_values = core.calc_eigenvalues()

    # nn.set_input(eigen_values)
    # nn.set_num_hidden_layers(num_layers)
    # nn.set_num_hidden_nodes(num_nodes)
    # nn.set_num_in_nodes(num_in_nodes)
    # nn.set_num_out_nodes(num_out_nodes)
    # nn.learning_rate(learning_rate)
    #
    # output = nn.train()
    # predict_tau = nn.calc_output(output)
    
#    S, R = core.calc_S_R(grad_u, k, eps)
#    input = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
#    T = core.calc_T(S, R)  # Tensor basis T
#    b = core.calc_output(stresses, k)  # b vector from tau tensor
#
#    print(input.shape)
#    num_inputs = input.shape[1]
#    num_outputs = b.shape[1]
#    num_layers = 8
#    num_nodes = 30
#
#    nn = NN(num_layers, num_nodes, num_inputs, num_outputs)


    # plot_results(b, b)
    
main()
