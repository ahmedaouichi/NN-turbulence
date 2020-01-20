import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m
from collections import defaultdict

class Calculator:

    def calc_S_R_test(grad_u, k, eps):
        n = grad_u.shape[0]
        S = np.zeros((n, 3, 3))
        R = np.zeros((n, 3, 3))
        ke = k / eps
        for i in range(n):
            S[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            R[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        return S,R

    def calc_scalar_basis_test(S, R):
        num_points = S.shape[0]
        num_eigenvalues = 5
        eigenvalues = np.zeros((num_points, num_eigenvalues))
        for i in range(num_points):
            eigenvalues[i, 0] = np.trace(np.dot(S[i, :, :], S[i, :, :]))
            eigenvalues[i, 1] = np.trace(np.dot(R[i, :, :], R[i, :, :]))
            eigenvalues[i, 2] = np.trace(np.dot(S[i, :, :], np.dot(S[i, :, :], S[i, :, :])))
            eigenvalues[i, 3] = np.trace(np.dot(R[i, :, :], np.dot(R[i, :, :], S[i, :, :])))
            eigenvalues[i, 4] = np.trace(np.dot(np.dot(R[i, :, :], R[i, :, :]), np.dot(S[i, :, :], S[i, :, :])))

        return eigenvalues


    def calc_S_R(grad_u, k, eps):
        DIM_Y = grad_u.shape[0]
        DIM_Z = grad_u.shape[1]

        S = np.zeros((DIM_Y, DIM_Z, 3, 3))
        R = np.zeros((DIM_Y, DIM_Z, 3, 3))
        ke = k / eps

        for ii in range(DIM_Y):
            for jj in range(DIM_Z):
                S[ii, jj, :, :] = ke[ii, jj] * 0.5 * (grad_u[ii, jj, :, :] + np.transpose(grad_u[ii, jj,  :, :]))
                R[ii, jj, :, :] = ke[ii, jj] * 0.5 * (grad_u[ii, jj, :, :] - np.transpose(grad_u[ii, jj, :,  :]))

        return S,R

    def calc_tensor_basis(S, R):
        DIM_Y = S.shape[0]
        DIM_Z = S.shape[1]

        T = np.zeros((DIM_Y, DIM_Z, 10, 3, 3))
        for ii in range(DIM_Y):
            for jj in range(DIM_Z):
                sij = S[ii, jj, :, :]
                rij = R[ii, jj, :, :]
                T[ii, jj, 0, :, :] = sij
                T[ii, jj, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
                T[ii, jj, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
                T[ii, jj, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
                T[ii, jj, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
                T[ii, jj, 5, :, :] = np.dot(rij, np.dot(rij, sij)) + np.dot(sij, np.dot(rij, rij)) - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
                T[ii, jj, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
                T[ii, jj, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
                T[ii, jj, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) + np.dot(np.dot(sij, sij), np.dot(rij, rij))- 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
                T[ii, jj, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))

        return T

    def calc_scalar_basis(S, R):
        DIM_Y = R.shape[0]
        DIM_Z = R.shape[1]

        eigenvalues = np.zeros([DIM_Y, DIM_Z, 5])
        for ii in range(DIM_Y):
            for jj in range(DIM_Z):
                eigenvalues[ii, jj, 0] = np.trace(np.dot(S[ii, jj, :, :], S[ii, jj, :, :]))
                eigenvalues[ii, jj, 1] = np.trace(np.dot(R[ii, jj, :, :], R[ii, jj, :, :]))
                eigenvalues[ii, jj, 2] = np.trace(np.dot(S[ii, jj, :, :], np.dot(S[ii, jj, :, :], S[ii, jj, :, :])))
                eigenvalues[ii, jj, 3] = np.trace(np.dot(R[ii, jj, :, :], np.dot(R[ii, jj, :, :], S[ii, jj, :, :])))
                eigenvalues[ii, jj, 4] = np.trace(np.dot(np.dot(R[ii, jj, :, :], R[ii, jj, :, :]), np.dot(S[ii, jj, :, :], S[ii, jj, :, :])))

        return eigenvalues

    def calc_output(tau, k):

        num_points = tau.shape[0]
        b = np.zeros((num_points, 3, 3))

        for i in range(3):
            for j in range(3):
                b[:, i, j] = tau[:, i, j]/(2.0*k)
            b[:, i, i] -= 1./3.

        b = np.reshape(b, (-1, 9))

        return b

    def calc_tensor(b, k):

        num_points = b.shape[0]
        tau = np.zeros((num_points, 3, 3))
        b = np.reshape(b, (-1, 3, 3))

        for i in range(3):
            tau[:, i, i] = 2.0*k*(b[:, i, i]+1./3.)
            for j in range(3):
                if(j!=i):
                    tau[:, i, j] = b[:, i, j]*2.0*k

        tau = np.reshape(tau, (-1, 9))

        return tau

    def calc_k(tau):
        k = 0.5 * (tau[:, 0, 0] + tau[:, 1, 1] + tau[:, 2, 2])
        k = np.maximum(k, 1e-8)
        return k
