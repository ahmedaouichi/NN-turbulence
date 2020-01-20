from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
import keras.backend as K
from keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Activation, BatchNormalization, Bidirectional,
                                     Conv1D, Dense, Dropout, Input, Lambda, Masking,
                                     TimeDistributed)

class NN:

    def __init__(self, num_layers, num_nodes, num_inputs, num_outputs, output_shape):
        self.num_layers = num_layers  # Number of hidden layers
        self.num_nodes = num_nodes  # Number of nodes per hidden layer
        self.num_inputs = num_inputs  # Number of scalar invariants
        self.num_outputs = num_outputs  # Number of tensors in the tensor basis
        self.output_shape = output_shape # Shape of b

    def build(self, dim):

        g_input = Input(shape=(self.num_inputs,))
        hidden = Dense(self.num_nodes, activation='relu')(g_input)
        for i in range(self.num_layers-1):
            hidden = Dense(self.num_nodes, activation='relu')(hidden)
        g_output = Dense(self.num_outputs, activation='relu')(hidden)
        tensor_in = Input(shape=(self.num_outputs, 9,))
        merge_layer = MergeLayer(dim)([tensor_in, g_output])

        merge_layer = BatchNormalization(axis=-1, momentum=0.5, epsilon=1e-3, \
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', \
        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, \
        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(merge_layer)

        model = Model(inputs=[tensor_in, g_input], outputs=merge_layer)
        # optimizer = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        optimizer = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        model.compile(loss = 'mean_squared_error' , optimizer = optimizer, metrics = ['accuracy'])

        self.model = model

    def train(self, invariants, tb, b):
        plot_model(self.model,show_shapes=True, expand_nested=True, to_file='model.png')
        self.model.fit([tb, invariants], b, batch_size=64, epochs=2, verbose=1)

class MergeLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MergeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        g = inputs[1]
        t = inputs[0]
        output = K.batch_dot(g, t, axes=[1,1])
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (None, 9)
        return output_shape
