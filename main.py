import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# from core import Core
from nn import NN

def main():
    # core = Core()
    # core.loadData('../inversion/DATA/SQUAREDUCT/DATA/03500_full.csv')
    # tau = core.get_tau()
    # u = core.get_u()
    # grad_u = core.calc_gradient()
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
