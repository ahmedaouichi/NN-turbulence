import matplotlib.pyplot as plt
import numpy as np
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

    k, eps, grad_u, stresses = load_test_data()

    core = Core()
    S, R = core.calc_S_R(grad_u, k, eps)
    input = core.calc_scalar_basis(S, R)  # Scalar basis lamba's
    T = core.calc_T(S, R)  # Tensor basis T
    b = core.calc_output(stresses, k)  # b vector from tau tensor

    nn = NN(8, 30, input.shape[1], T.shape[1], b.shape[1])
    result = nn.build(input, T, b)

    plot_results(result, b)


main()
