import numpy as np

class NN:

    def __init__(self, num_layers, num_nodes, num_inputs, num_outputs):
        self.num_layers = num_layers  # Number of hidden layers
        self.num_nodes = num_nodes  # Number of nodes per hidden layer
        self.num_inputs = num_inputs  # Number of scalar invariants
        self.num_outputs = num_outputs  # Number of tensors in the tensor basis


    
