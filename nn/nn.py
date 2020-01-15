import keras
from keras import optimizers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model

class NN:

    def __init__(self, num_layers, num_nodes, num_inputs, num_outputs, output_shape):
        self.num_layers = num_layers  # Number of hidden layers
        self.num_nodes = num_nodes  # Number of nodes per hidden layer
        self.num_inputs = num_inputs  # Number of scalar invariants
        self.num_outputs = num_outputs  # Number of tensors in the tensor basis
        self.output_shape = output_shape # Shape of b

    def build(self):
        g_input = keras.layers.Input(shape=(self.num_inputs,))
        hidden = keras.layers.Dense(self.num_nodes, activation='sigmoid')(g_input)
        for i in range(self.num_layers-1):
            hidden = keras.layers.Dense(self.num_nodes, activation='sigmoid')(hidden)
        g_output = keras.layers.Dense(self.num_outputs, activation='sigmoid')(hidden)
        tensor_in = keras.layers.Input(shape=(self.num_outputs, 9,))
        merge_layer = keras.layers.Dot([1,1], normalize=True)([tensor_in, g_output])
        normalize_layer = keras.layers.BatchNormalization(axis=-1, momentum=0.5, epsilon=1e-3, \
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', \
        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, \
        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(merge_layer)
        model = keras.models.Model(inputs=[tensor_in, g_input], outputs=normalize_layer)
        # optimizer = keras.optimizers.RMSprop(learning_rate=0.09)
        optimizer = optimizers.SGD(lr=1, decay=1e-6, momentum=0.5, nesterov=True)
        # optimizer = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['accuracy'])
        self.model = model

    def train(self, invariants, tb, b):
        history = self.model.fit([tb, invariants], b, batch_size = 200, nb_epoch = 10, verbose = 1)
        # SVG(model_to_dot(self.model).create(prog='dot', format='svg'))
        plot_model(self.model,show_shapes=True, expand_nested=True, to_file='model.png')
#        result = self.model.predict([tb, invariants])
#        SVG(model_to_dot(self.model, show_shapes=True, show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))
#        print(self.model.summary())
#        return result
