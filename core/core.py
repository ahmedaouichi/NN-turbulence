import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl


class Core:

    object_counter = 0

    def __init__(self):
        self.object_id = Core.object_counter
        Core.object_counter += 1

    #### Input: path to file relative to core.py file
    #### Output:
    def loadData(self, filepath):

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            names = next(reader)
            ubulk = next(reader)
            print(names)
            data_list = np.zeros([len(names), 100000])
            for i,row in enumerate(reader):
                if row:
                    data_list[:,i] = np.array([float(ii) for ii in row])

        data_list = data_list[:,:i+1]
        data = {}
        for j,var in enumerate(names):
            data[var] = data_list[j,:]

        self.data = data

        return self.data

    def calc_S_R(self, grad_u, k, eps):
        n = grad_u.shape[0]
        S = np.zeros((n, 3, 3))
        R = np.zeros((n, 3, 3))
        ke = k / eps
        for i in range(n):
            S[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            R[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        return S,R

    def calc_T(self, S, R):

        num_points = S.shape[0]
        num_tensor_basis = 10
        T = np.zeros((num_points, num_tensor_basis, 3, 3))
        for i in range(num_points):
            sij = S[i, :, :]
            rij = R[i, :, :]
            T[i, 0, :, :] = sij
            T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
            T[i, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
            T[i, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
            T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
            T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) + np.dot(sij, np.dot(rij, rij)) - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
            T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
            T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
            T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) + np.dot(np.dot(sij, sij), np.dot(rij, rij))- 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
            T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))

        return T

    def calc_scalar_basis(self, S, R):
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

    def calc_output(self, tau, k):

        num_points = tau.shape[0]
        b = np.zeros((num_points, 3, 3))

        for i in range(3):
            for j in range(3):
                b[:, i, j] = tau[:, i, j]/(2.0 * k)
            b[:, i, i] -= 1./3.

        b_vector = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                b_vector[:, 3*i+j] = b[:, i, j]
        return b_vector

    def partialderivative(u0, u1, u2, q0, q2):
        return (u0-2*u1+u2)/(q2-q0)

    def partialderivativeFD(u1, u2, q1, q2):
        return (u2-u1)/(q2-q1)

    def partialderivativeBD(u0, u1, q0, q1):
        return (u1-u0)/(q1-q0)

    def calc_gradient(self):
        data = self.data
        grid_size = int(len(data['Y'])**0.5)

        yy = np.reshape(data['Y'], [grid_size,grid_size])
        zz = np.reshape(data['Z'], [grid_size,grid_size])
        uu = np.reshape(data['um'],[grid_size,grid_size])
        vv = np.reshape(data['vm'],[grid_size,grid_size])
        ww = np.reshape(data['wm'],[grid_size,grid_size])

        ## Create matrix containing u,v and w for every (x,y) data point
        ## Matrix[X, Y, 3 (u,v,w)]
        u = np.array([uu,vv,ww])

        ## Calculate gradient using numpy-function. Returns three lists,
        ## each containing an array corresponding to the derivative to one dimension.

        gradient_manual = np.zeros([grid_size,grid_size,3,3])

        for ii in range(1,grid_size-1):
            for jj in range(1,grid_size-1):
                ## As the flow is symmetric in the x-direction, the gradient in x-direction is zero.
                ux_x = 0
                uy_x = 0
                uz_x = 0

                y0 = yy[ii-1,jj]
                y1 = yy[ii,jj]
                y2 = yy[ii+1,jj]

                z0 = zz[ii,jj-1]
                z1 = zz[ii,jj]
                z2 = zz[ii,jj+1]

                u_y0 = uu[ii-1,jj]
                u_y1 = uu[ii,jj]
                u_y2 = uu[ii+1,jj]

                u_z0 = uu[ii,jj-1]
                u_z1 = uu[ii,jj]
                u_z2 = uu[ii,jj-1]

                v_y0 = vv[ii-1,jj]
                v_y1 = vv[ii,jj]
                v_y2 = vv[ii+1,jj]

                v_z0 = vv[ii,jj-1]
                v_z1 = vv[ii,jj]
                v_z2 = vv[ii,jj+1]

                w_y0 = ww[ii-1,jj]
                w_y1 = ww[ii,jj]
                w_y2 = ww[ii+1,jj]

                w_z0 = ww[ii,jj-1]
                w_z1 = ww[ii,jj]
                w_z2 = ww[ii,jj+1]


                ux_y = Core.partialderivative(u_y0, u_y1, u_y2, y0, y2)
                uy_y = Core.partialderivative(v_y0, v_y1, v_y2, y0, y2)
                uz_y = Core.partialderivative(w_y0, w_y1, w_y2, y0, y2)

                ux_z = Core.partialderivative(u_z0, u_z1, u_z2, z0, z2)
                uy_z = Core.partialderivative(v_z0, v_z1, v_z2, z0, z2)
                uz_z = Core.partialderivative(w_z0, w_z1, w_z2, z0, z2)

                gradient_tensor = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
                gradient_manual[ii,jj,:,:] = gradient_tensor


        ####### THIS NEED TO BE FINISHED!
#        ## Calculate tensors for boundary points
#        for jj in range(1,grid_size-1):
#            ii = 0
#            ux_y = Core.partialderivativeFD(u_y1, u_y2, y1, y2)
#            uy_y = Core.partialderivativeFD(v_y1, v_y2, y1, y2)
#            uz_y = Core.partialderivativeFD(w_y1, w_y2, y1, y2)
#
#        if (ii == grid_size-1):
#            ux_y = Core.partialderivativeBD(u_y0, u_y1, y0, y1)
#            uy_y = Core.partialderivativeBD(v_y0, v_y1, y0, y1)
#            uz_y = Core.partialderivativeBD(w_y0, w_y1, y0, y1)
#
        gradient_tensor

        # print(gradient_manual[1,1,:,:])

        gradient = np.zeros([129,129,3,3])
        for ii in range(0):
            for jj in range(0):
                for nn in range(3):
                    for mm in range(3):
                        gradient[ii,jj,nn,mm] = nabla_u[nn][mm,ii,jj]
