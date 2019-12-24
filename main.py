import numpy as np
from nn import NN
from core import Core
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import keras.backend as K
import math as m

def a(x,y):
    return x**2*y**2

def b(x,y):
    return x*y

def c(x,y):
    return y**3*x**3

#
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
            if i in on_diag:
                plt.xlim([-1./3., 2./3.])
                plt.ylim([-1./3., 2./3.])
            else:
                plt.xlim([-0.5, 0.5])
                plt.ylim([-0.5, 0.5])
    plt.show()

def main():
    core = Core()

    ## Specify usecase: (RA)_(Retau), RA = ratio aspect, Retau is usecase = '3_180'
    eigenvalues_shape = 5
    tensorbasis_shape = 10
    stresstensor_shape = 9

    Retau = 180
    RA_list = [1,3,5,7,10,14]
    velocity_comps = ['U', 'V', 'W']
    DIM = len(velocity_comps)


    print('--> Build network')
    neural_network = NN(2, 20, eigenvalues_shape, tensorbasis_shape, stresstensor_shape)
    neural_network.build()

    RA = RA_list[0]
    usecase =  str(RA)+'_'+str(Retau)

    print('--> Import coordinate system')
    ycoord, DIM_Y = Core.importCoordinates('y', usecase)
    zcoord, DIM_Z = Core.importCoordinates('z', usecase)

    print('--> Import mean velocity field')
    data = np.zeros([DIM_Y, DIM_Z, DIM])
    for ii in range(DIM):
        data[:,:,ii] = core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_comps[ii])

    print('--> Compute mean velocity gradient')
    velocity_gradient = core.gradient(data, ycoord, zcoord)
    print('--> Import Reynolds stress tensor')
    stresstensor = core.importStressTensor(usecase, DIM_Y, DIM_Z)
    stresstensor = np.reshape(stresstensor, (-1, 3, 3))

    ############ Preparing network inputs #####################################

    print('--> Compute omega field')
    eps = 1*np.ones([DIM_Y, DIM_Z]) ## For now using epsilon=1 everywhere
    print('--> Compute k field')
    k = core.calc_k(stresstensor)
    k = np.reshape(k, (DIM_Y, DIM_Z))
    print('--> Compute rotation rate and strain rate tensors')
    S,R = core.calc_S_R(velocity_gradient, k, eps)
    print('--> Compute eigenvalues for network input')
    eigenvalues = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
    print('--> Compute tensor basis')
    tensorbasis = core.calc_tensor_basis(S, R)


    ## Reshape 2D array to 1D arrays. Network only takes 1D arrays
    k = np.reshape(k, -1)
    b = core.calc_output(stresstensor, k)
    stresstensor = np.reshape(stresstensor, (-1, 9))
    tensorbasis = np.reshape(tensorbasis, (-1, 10, 9))
    eigenvalues = np.reshape(eigenvalues, (-1, 5))
    b = np.reshape(b, (-1, 9))

    print('--> Build network')
    neural_network = NN(8, 30, eigenvalues.shape[1], tensorbasis.shape[1], b.shape[1])
    neural_network.build()


    print('--> Train network')
    prediction = neural_network.train(eigenvalues, tensorbasis, b)

    plot_results(prediction, b)

    core.tensorplot(stresstensor, DIM_Y, DIM_Z)
    predicted_stress = core.calc_tensor(prediction, k)
    core.tensorplot(predicted_stress, DIM_Y, DIM_Z)

    ###############  Visualize prediction  ####################################

#    from keras.utils.vis_utils import plot_model
#
#    plot_model(neural_network.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    plt.show()

main()


# def load_test_data():
#     data = np.loadtxt("nn/test_data.txt", skiprows=4)
#     k = data[:, 0]
#     eps = data[:, 1]
#     grad_u_flat = data[:, 2:11]
#     stresses_flat = data[:, 11:]
#
#     num_points = data.shape[0]
#     grad_u = np.zeros((num_points, 3, 3))
#     stresses = np.zeros((num_points, 3, 3))
#     for i in range(3):
#         for j in range(3):
#             grad_u[:, i, j] = grad_u_flat[:, i*3+j]
#             stresses[:, i, j] = stresses_flat[:, i*3+j]
#
#     return k, eps, grad_u, stresses

#
# def make_realizable(labels):
#         numPoints = labels.shape[0]
#         A = np.zeros((3, 3))
#         for i in range(numPoints):
#             # Scales all on-diags to retain zero trace
#             if np.min(labels[i, [0, 4, 8]]) < -1./3.:
#                 labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
#             if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
#                 labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
#                 labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
#             if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
#                 labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
#                 labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
#             if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
#                 labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
#                 labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
#
#             # Enforce positive semidefinite by pushing evalues to non-negative
#             A[0, 0] = labels[i, 0]
#             A[1, 1] = labels[i, 4]
#             A[2, 2] = labels[i, 8]
#             A[0, 1] = labels[i, 1]
#             A[1, 0] = labels[i, 1]
#             A[1, 2] = labels[i, 5]
#             A[2, 1] = labels[i, 5]
#             A[0, 2] = labels[i, 2]
#             A[2, 0] = labels[i, 2]
#             evalues, evectors = np.linalg.eig(A)
#             if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/2.:
#                 evalues = evalues*(3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/(2.*np.max(evalues))
#                 A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
#                 for j in range(3):
#                     labels[i, j] = A[j, j]
#                 labels[i, 1] = A[0, 1]
#                 labels[i, 5] = A[1, 2]
#                 labels[i, 2] = A[0, 2]
#                 labels[i, 3] = A[0, 1]
#                 labels[i, 7] = A[1, 2]
#                 labels[i, 6] = A[0, 2]
#             if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
#                 evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
#                 A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
#                 for j in range(3):
#                     labels[i, j] = A[j, j]
#                 labels[i, 1] = A[0, 1]
#                 labels[i, 5] = A[1, 2]
#                 labels[i, 2] = A[0, 2]
#                 labels[i, 3] = A[0, 1]
#                 labels[i, 7] = A[1, 2]
#                 labels[i, 6] = A[0, 2]
#
#         return labels
#
# def plot_tensors(predicted, true):
#
#     predicted.reshape()
#
#
# def main():
#
#     core = Core()
#     k, eps, grad_u, stresses = load_test_data()
#     S, R = core.calc_S_R_test(grad_u, k, eps)
#     input = core.calc_scalar_basis_test(S, R)  # Scalar basis lamba's
#     T = core.calc_T_test(S, R)  # Tensor basis T
#     b = core.calc_output(stresses, k)  # b vector from tau tensor
#     nn = NN(2, 20, input.shape[1], T.shape[1], b.shape[1])
#     nn.build()
#
#     print(input.shape)
#     print(T.shape)
#     print(b.shape)
#     print(stresses.shape)
#
#     # for i in range(50):
#     #         b = make_realizable(b)
#
#     result = nn.train(input, T, b)
#
#     # for i in range(50):
#     #         result = make_realizable(result)
#
#     # plot_tensors(result, b)
#     plot_results(result, b)
#
# main()
