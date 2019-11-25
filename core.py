import numpy as np


class Calculator:

    def __init__(self, x):
        self.x = x

    def calc(self):
        self.x += 1

    def calc_Sij_Rij(grad_u, k, eps, n):

        S = np.zeros((n, 3, 3))
        R = np.zeros((n, 3, 3))
        for i in xrange(n):
            S[i, :, :] = k[i]/eps[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            R[i, :, :] = k[i]/eps[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))
