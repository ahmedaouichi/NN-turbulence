import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt


class NN:

    def __init__(self, num_layers, num_nodes, num_inputs, num_outputs, output_shape):
        self.num_layers = num_layers  # Number of hidden layers
        self.num_nodes = num_nodes  # Number of nodes per hidden layer
        self.num_inputs = num_inputs  # Number of scalar invariants
        self.num_outputs = num_outputs  # Number of tensors in the tensor basis
        self.output_shape = output_shape # Shape of b

    def build(self, invariants, tb, b):
        g_input = keras.layers.Input(shape=(self.num_inputs,))
        hidden = keras.layers.Dense(self.num_nodes, activation='relu')(g_input)
        for i in range(self.num_layers-1):
            hidden = keras.layers.Dense(self.num_nodes, activation='relu')(hidden)
        g_output = keras.layers.Dense(self.num_outputs, activation='relu')(hidden)

        tensor_in = keras.layers.Input(shape=(self.num_outputs, 9,))
         # T.batched_tensordot(inputs[0], inputs[1], axes=[[1], [1]])
        multiply_layer = keras.layers.Dot([1,1], normalize=False)([tensor_in, g_output])
        b_output = keras.layers.Dense(9)(multiply_layer)

        model = keras.models.Model(inputs=[tensor_in, g_input], outputs=b_output)
        model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])
        model.fit([tb, invariants], b, batch_size = 200000, epochs = 100, verbose = 0)
        self.model = model
    
    def plot_results(self,predicted_stresses, true_stresses):
    
        components = ['uu', 'vu', 'wu', 'uv', 'vv', 'wv', 'uw', 'vw', 'ww']
        
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
                plt.title(components[i])
                if i in on_diag:
                    plt.xlim([-1e-2, 1e-2])
                    plt.ylim([-1e-2, 1e-2])
                else:
                    plt.xlim([-1e-2, 1e-2])
                    plt.ylim([-1e-2, 1e-2])
        plt.tight_layout()
        plt.show()