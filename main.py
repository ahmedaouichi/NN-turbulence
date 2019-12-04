import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
from core import Core
from nn import NN
import math as m


def main():
    core = Core()
    ############# Rectangular Duct data ########################
    ## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'
    
    Retau = 180
    RA = [1,3,5,7,10,14]
    
    ## 3rd input argument refers to one of velocity components <U, V, W>
    core.plotMeanVelocity(RA, Retau, 'U')
    
    
    ##core.loadData('../inversion/DATA/SQUAREDUCT/DATA/03500_full.csv')
    # tau = core.get_tau()
    # u = core.get_u()
    ## grad_u = core.calc_gradient() 
    # k = core.calc_k()
    # eps = core.calc_epsilon()
    # n = core.get_n()
    # S,R = core.calc_S_R(grad_u, k, eps, n)
    # eigen_values = core.calc_eigenvalues()

    nn = NN()
    
    # nn.set_input(eigen_values)
    # nn.set_num_hidden_layers(num_layers)
    # nn.set_num_hidden_nodes(num_nodes)
    # nn.set_num_in_nodes(num_in_nodes)
    # nn.set_num_out_nodes(num_out_nodes)
    # nn.learning_rate(learning_rate)
    #
    # output = nn.train()
    # predict_tau = nn.calc_output(output)
    
main()
