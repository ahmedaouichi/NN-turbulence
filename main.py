import matplotlib.pyplot as plt
import numpy as np
from nn import NN
from core import Core
from keras.models import Sequential
from keras.layers import Dense, Lambda
import keras.backend as K

def load_test_data():
    data = np.loadtxt("nn/test_data.txt", skiprows=4)
    k = data[:, 0]
    eps = data[:, 1]
    grad_u_flat = data[:, 2:11]
    stresses_flat = data[:, 11:]

    num_points = data.shape[0]
    grad_u = np.zeros((num_points, 3, 3))
    stresses = np.zeros((num_points, 3, 3))
    for i in xrange(3):
        for j in xrange(3):
            grad_u[:, i, j] = grad_u_flat[:, i*3+j]
            stresses[:, i, j] = stresses_flat[:, i*3+j]

    return k, eps, grad_u, stresses

def plot_results(predicted_stresses, true_stresses):

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    on_diag = [0, 4, 8]
    for i in xrange(9):
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

    core = Core()
    k, eps, grad_u, stresses = load_test_data()
    S, R = core.calc_S_R_test(grad_u, k, eps)
    input = core.calc_scalar_basis_test(S, R)  # Scalar basis lamba's
    T = core.calc_T_test(S, R)  # Tensor basis T
    b = core.calc_output(stresses, k)  # b vector from tau tensor
    nn = NN(8, 30, input.shape[1], T.shape[1], b.shape[1])
    result = nn.build(input, T, b)

    plot_results(result, b)

    ############# Rectangular Duct data #######################################

    ## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'

#     Retau = 180
#     RA_list = [1,3,5,7,10,14]
#     velocity_comps = ['U', 'V', 'W']
#     DIM = len(velocity_comps)
#
#     RA = RA_list[0]
#     usecase =  str(RA)+'_'+str(Retau)
#
#     print('--> Import coordinate system')
#     ycoord, DIM_Y = Core.importCoordinates('y', usecase)
#     zcoord, DIM_Z = Core.importCoordinates('z', usecase)
#
#     print('--> Import mean velocity field')
#     data = np.zeros([DIM_Y, DIM_Z, DIM])
#     for ii in range(DIM):
#         data[:,:,ii] = core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_comps[ii])
#
#
#     print('--> Compute mean velocity gradient')
#     velocity_gradient = core.gradient(data, ycoord, zcoord)
#     print('--> Import Reynolds stress tensor')
#     tensor = core.importStressTensor(usecase, DIM_Y, DIM_Z)
#
#     ############# Visualization ###############################################
#
#     ## 3rd input argument refers to one of velocity components <U, V, W>
#     #core.plotMeanVelocityComponent(RA_list, Retau, 'U')
#
#     #core.plotMeanVelocityField(RA, Retau, data, ycoord, zcoord)
#
#     ############ Preparing network inputs #####################################
#
#     print('--> Compute omega field')
#     eps = np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere
#     print('--> Compute k field')
#     k = core.calc_k(velocity_gradient)
#     print('--> Compute rotation rate and strain rate tensors')
#     S,R = core.calc_S_R(velocity_gradient, k, eps)
#     print('--> Compute eigenvalues for network input')
#     eigenvalues = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
#     print('--> Compute tensor basis')
#     #tensorbasis = core.calc_tensor_basis(S, R)
#     #print(np.shape(tensorbasis))
#
#
#     ############  Building neural network  ####################################
#
#     number_layers = 8
#     number_nodes = 30
# #
# #
#     def customLoss(yTrue,yPred):
#         return K.sum((yTrue - yPred)**2)**0.5
#
#     input_size = DIM_Y*DIM_Z
#     print(np.shape(eigenvalues))
#     X = np.reshape(eigenvalues, [input_size, 5]) ## rearange 2d matrix to 1d matrix
#     Y = np.tile(np.arange(0,10), (input_size, 1)) ## Here we need a list of g values, which we don't know how to obtain them
#
#     # define the keras model
#     model = Sequential()
#     model.add(Dense(30, input_dim=5, activation='relu'))
#     model.add(Dense(30, activation='relu'))
#     model.add(Dense(30, activation='relu'))
#     model.add(Dense(30, activation='relu'))
#     model.add(Dense(30, activation='relu'))
#     model.add(Dense(30, activation='relu'))
#     model.add(Dense(30, activation='relu'))
#     model.add(Dense(10, activation='relu'))
#
#     # compile the keras model
#     model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy'])
#
#     # fit the keras model on the dataset
#     model.fit(X, Y, epochs=2, batch_size=10)
#
#     # evaluate the keras model
#     _, accuracy = model.evaluate(X, Y)
#     print('Accuracy: %.2f' % (accuracy*100))


main()
