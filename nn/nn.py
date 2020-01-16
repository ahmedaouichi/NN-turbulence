import keras
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

        # def calc(x):
        #     (a, b) = pywt.dwt(x, 'db1')
        #     return K.dot(a,b)

        def my_loss(y_true, y_pred):
            loss=K.mean(K.sum(K.square(y_true-y_pred)))
            return loss

        g_input = keras.layers.Input(shape=(self.num_inputs,))
        hidden = keras.layers.Dense(self.num_nodes, activation='softmax')(g_input)
        for i in range(self.num_layers-1):
            hidden = keras.layers.Dense(self.num_nodes, activation='softmax')(hidden)
        g_output = keras.layers.Dense(self.num_outputs, activation='softmax')(hidden)
        tensor_in = keras.layers.Input(shape=(self.num_outputs, 9,))

        # tensor_in = Lambda(lambda x: tf.stop_gradient(x))(tensor_in)
        # g_output = Lambda(lambda x: tf.stop_gradient(x))(g_output)

        # merge_layer = keras.layers.Dot([1,1], normalize=True)([tensor_in, g_output])
        merge_layer = MergeLayer(dim)([tensor_in, g_output])

        # g_output = Lambda(lambda x: tf.stop_gradient(x))(g_output)

        # merge_layer = keras.layers.Multiply()([tensor_in, g_output])
        # merge_layer = keras.layers.Lambda(calc, output_shape=(9,))([g_output, tensor_in])
        # merge_layer = keras.layers.Flatten(data_format=None)(merge_layer)

        # normalize_layer = keras.layers.BatchNormalization(axis=1, momentum=0.5, epsilon=1e-3, \
        # center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', \
        # moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, \
        # gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(merge_layer)
        model = keras.models.Model(inputs=[tensor_in, g_input], outputs=merge_layer)
        # optimizer = keras.optimizers.RMSprop(learning_rate=0.09)
        # optimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
        optimizer = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(loss = 'mean_squared_error' , optimizer = optimizer, metrics = ['accuracy'])
        # Show summary
        # model.summary()
        self.model = model

    def train(self, invariants, tb, b):
        print(tb.shape)
        print(invariants.shape)
        plot_model(self.model,show_shapes=True, expand_nested=True, to_file='model.png')
        history = self.model.fit([tb, invariants], b, batch_size = 200, epochs= 20, verbose = 1)
        # SVG(model_to_dot(self.model).create(prog='dot', format='svg'))
        result = self.model.predict([tb, invariants])
        # print(self.model.summary())
        return result

class MergeLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        # self.total = tf.Variable(initial_value=tf.zeros((self.output_dim,9,)), trainable=False)
        super(MergeLayer, self).__init__(**kwargs)

    # def build(self, input_shape):

        # self.total = tf.Variable(initial_value=tf.zeros((self.output_dim,9,)), trainable=False)

        # super(MergeLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        g = inputs[1]
        t = inputs[0]

        output = tf.zeros((self.output_dim,9,))

        output = K.batch_dot(g, t, axes=[1,1])
        # self.total.assign_add(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (None, 9)
        return output_shape
