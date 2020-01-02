import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m
from collections import defaultdict

class Core:

    object_counter = 0

    def __init__(self):
        self.object_id = Core.object_counter
        Core.object_counter += 1

    def importCoordinates(coordinate, usecase): ##### Input coordinate should be 'y' or 'z'
        ## As for RA=1 the grid is square, there is no file containing y coordinates for RA.
        ## In this case, y and z coordinates are identical.
        RA = int(usecase.split('_')[0])

        if (coordinate == 'y'):
            if (RA == 1): ## In case of a square grid, y and z are identical
                coordinate = 'z'

        ## Open appropriate file and read it out
        dataline = []
        filepath = '../inversion/DATA/RECTANGULARDUCT/DATA/'+coordinate+'coord_' + usecase + '.prof.txt'
        data = []
        with open(filepath, 'r', encoding='latin-1') as fd:
            for line in fd:
                dataline = line.split()
                if len(dataline) != 0:
                    data = dataline

            DIM = len(data)

        fd.close()

        coord = np.zeros(DIM)
        for ii in range(DIM-1):
            coord[ii] = data[ii]
        return coord, DIM

    def importMeanVelocity(self, DIM_Y, DIM_Z, usecase, velocity_component):
        filepath = '../inversion/DATA/RECTANGULARDUCT/DATA/'+str(velocity_component)+'_' + str(usecase) + '.prof.txt'
        U = np.zeros([DIM_Y, DIM_Z])

        counter = 0
        with open(filepath, encoding='latin-1') as fd:
            for i in range(24):
                next(fd)
            for line in fd:
                dataline= line.split()
                if len(dataline) == DIM_Z: ## Omit blank lines
                    U[counter,:] = dataline
                    counter += 1

        fd.close()

        return U

    def importStressTensor(self, usecase, DIM_Y, DIM_Z):
        ## Open appropriate file and read it out

        components = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
        data = defaultdict(list)
        for ii in range(len(components)):
            comp = components[ii]
            filepath = '../inversion/DATA/RECTANGULARDUCT/DATA/'+comp+ '_' + usecase + '.prof.txt'
            data[comp]  = []
            with open(filepath, 'r', encoding='latin-1') as fd:
                for line in fd:
                    if line[0] != '%':
                        data[comp].append(line.split())
                        break

                for line in fd:
                    if len(line.split()) != 0: ## Omit blank lines
                        data[comp].append(line.split())

            fd.close()

        tensor = np.ones([DIM_Y, DIM_Z, 3,3])
        counter = 1
        for ii in range(DIM_Y):
            for jj in range(DIM_Z):
                tensor[ii,:,0,0] = data['uu'][ii]
                tensor[ii,:,1,0] = data['uv'][ii]
                tensor[ii,:,2,0] = data['uw'][ii]

                tensor[ii,:,0,1] = data['uv'][ii]
                tensor[ii,:,1,1] = data['vv'][ii]
                tensor[ii,:,2,1] = data['vw'][ii]

                tensor[ii,:,0,2] = data['uw'][ii]
                tensor[ii,:,1,2] = data['vw'][ii]
                tensor[ii,:,2,2] = data['ww'][ii]

                counter += 1

        return tensor

    def plotMeanVelocityComponent(self, RA, Retau, velocity_component):

        fig = plt.figure(1,figsize=(10,30))
        gs = fig.add_gridspec(6, 14)
        fig.suptitle('Mean velocity flow for $R_{\\tau}$ and different RA values for velocity compoment '+ velocity_component)
        fig.subplots_adjust(hspace = 1)
        for ii in range(len(RA)):
            ra = RA[ii]
            ncols = RA[ii]-1

            ### Collect coordinates and mean velocity data
            usecase = str(ra)+'_'+str(Retau)

            zcoord, DIM_Z = Core.importCoordinates('z', usecase)
            ycoord, DIM_Y = Core.importCoordinates('y', usecase)

            U = Core.importMeanVelocity(self, DIM_Y, DIM_Z, usecase, velocity_component)

            ### Create subplots
            if (ncols == 0):
                ax = fig.add_subplot(gs[ii, 0])
            else:
                ax = fig.add_subplot(gs[ii, 0:ncols])

            ### Add contour graphs to subplots
            ax.contourf(U, extent=(np.amin(zcoord), np.amax(zcoord), np.amin(ycoord), np.amax(ycoord)), cmap=plt.cm.Oranges)
            ax.set_title('RA = '+str(ra))

            if (ii == m.ceil(len(RA)/2)):
                ax.set_ylabel('z: spanwise (a. u.)')

            if (ii == len(RA)-1):
                ax.set_xlabel('y: wall normal (a. u. )')

        plt.plot()

    def plotMeanVelocityField(self, RA, Retau, data, ycoord, zcoord):
        x, y, z = np.meshgrid(ycoord[1:-1:10], zcoord[1:-1:10], np.zeros(1))

        fig_field = plt.figure(2)
        ax = fig_field.gca(projection='3d')
        ax.quiver(x, y, z, data[1:-1:10,1:-1:10,0], data[1:-1:10,1:-1:10,1], data[1:-1:10,1:-1:10,2], length=0.05, normalize=True)
        ax.set_title('Mean Velocity field for $R_{\\tau}$ = '+str(Retau) +' and RA = '+str(RA))


    def calc_S_R_test(self, grad_u, k, eps):
        n = grad_u.shape[0]
        S = np.zeros((n, 3, 3))
        R = np.zeros((n, 3, 3))
        ke = k / eps
        for i in range(n):
            S[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            R[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        return S,R

    def calc_T_test(self, S, R):

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
            T_flat = np.zeros((num_points, num_tensor_basis, 9))
        for i in range(3):
            for j in range(3):
                T_flat[:, :, 3*i+j] = T[:, :, i, j]
        return T_flat

    def tensorplot(self, tensor, DIM_Y, DIM_Z):
        tensor = np.reshape(tensor, (DIM_Y, DIM_Z, 9))

        fig, axes = plt.subplots(nrows=3, ncols=3)
        ii = 0
        for ax in axes.flat:
            y = m.floor((ii-1)/3)
            z = int((ii-1)%3)
            im = ax.contourf(tensor[:,:,y*z])
            ii += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    def tensorplot(self, tensor, DIM_Y, DIM_Z):
        tensor = np.reshape(tensor, (DIM_Y, DIM_Z, 9))

        fig, axes = plt.subplots(nrows=3, ncols=3)
        ii = 0
        for ax in axes.flat:
            y = m.floor((ii-1)/3)
            z = int((ii-1)%3)
            im = ax.contourf(tensor[:,:,y*z])
            ii += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    def calc_scalar_basis_test(self, S, R):
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


    def calc_S_R(self, grad_u, k, eps):
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

    def calc_tensor_basis(self, S, R):
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

    def calc_scalar_basis(self, S, R):
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

    def calc_output(self, tau, k):

        num_points = tau.shape[0]
        b = np.zeros((num_points, 3, 3))

        for i in range(3):
            for j in range(3):
                tmp = 2.0*k
                b[:, i, j] = tau[:, i, j]/tmp
            b[:, i, i] -= 1./3.

        b_vector = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                b_vector[:, 3*i+j] = b[:, i, j]
        return b_vector

    def calc_tensor(self, b, k):

        num_points = b.shape[0]
        tau = np.zeros((num_points, 3, 3))
        b = np.reshape(b, (-1, 3, 3))

        for i in range(3):
            for j in range(3):
                tmp = 2.0*k
                tau[:, i, j] = b[:, i, j]*tmp
            tau[:, i, j] += 1./3.

        return tau

    def calc_k(self, tau):
        k = 0.5 * (tau[:, 0, 0] + tau[:, 1, 1] + tau[:, 2, 2])
        k = np.maximum(k, 1e-8)
        return k

    def partialderivativeCD(u0, u1, u2, q0, q2):
        return (u0-2*u1+u2)/(q2-q0)

    def partialderivativeFD(u1, u2, q1, q2):
        return (u2-u1)/(q2-q1)

    def partialderivativeBD(u0, u1, q0, q1):
        return (u1-u0)/(q1-q0)

    def coordinatesCD(uu,ii,jj, direction):
        if direction == 'y':
            return uu[ii-1,jj], uu[ii,jj], uu[ii+1,jj]
        else:
            return uu[ii,jj-1], uu[ii,jj], uu[ii,jj+1]

    def coordinatesFD(uu,ii,jj, direction):
        if direction == 'y':
            return uu[ii,jj], uu[ii+1,jj]
        else:
            return uu[ii,jj], uu[ii,jj+1]

    def coordinatesBD(uu,ii,jj, direction):
        if direction == 'y':
            return uu[ii-1,jj], uu[ii,jj]
        else:
            return uu[ii,jj-1], uu[ii,jj]

    def gradient(self,data,ycoord, zcoord):

        DIM_Y = len(ycoord)
        DIM_Z = len(zcoord)


        zz, yy = np.meshgrid(zcoord, ycoord)

        uu = data[:,:,0]
        vv = data[:,:,1]
        ww = data[:,:,2]

        ## Create matrix containing u,v and w for every (x,y) data point
        ## Matrix[X, Y, 3 (u,v,w)]
        u = np.array([uu,vv,ww])

        ## Calculate gradient using numpy-function. Returns three lists,
        ## each containing an array corresponding to the derivative to one dimension.

        gradient = np.zeros([DIM_Y,DIM_Z,3,3])

        ## As the flow is symmetric in the x-direction, the gradient in x-direction is always zero.
        ux_x = 0
        uy_x = 0
        uz_x = 0

        for ii in range(DIM_Y):
            for jj in range(DIM_Z):
                y1 = yy[ii,jj]
                z1 = zz[ii,jj]

                ## Inside grid points
                if ((ii > 0 and ii < DIM_Y-2) and (jj > 0 and jj < DIM_Z-2)):
                    y0 = yy[ii-1,jj]
                    y2 = yy[ii+1,jj]

                    z0 = zz[ii,jj-1]
                    z2 = zz[ii,jj+1]

                    u_y0, u_y1, u_y2 = Core.coordinatesCD(uu,ii,jj, 'y')
                    u_z0, u_z1, u_z2 = Core.coordinatesCD(uu,ii,jj, 'z')
                    v_y0, v_y1, v_y2 = Core.coordinatesCD(vv,ii,jj, 'y')
                    v_z0, v_z1, v_z2 = Core.coordinatesCD(vv,ii,jj, 'z')
                    w_y0, w_y1, w_y2 = Core.coordinatesCD(ww,ii,jj, 'y')
                    w_z0, w_z1, w_z2 = Core.coordinatesCD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
                    uy_y = Core.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
                    uz_y = Core.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)

                    ux_z = Core.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
                    uy_z = Core.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
                    uz_z = Core.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Upper boundary --> z = 0
                if ((ii > 0 and ii < DIM_Y-2) and (jj == 0)):
                    y0 = yy[ii-1,jj]
                    y2 = yy[ii+1,jj]

                    z2 = zz[ii,jj+1]

                    u_y0, u_y1, u_y2 = Core.coordinatesCD(uu,ii,jj, 'y')
                    u_z1, u_z2 = Core.coordinatesFD(uu,ii,jj, 'z')
                    v_y0, v_y1, v_y2 = Core.coordinatesCD(vv,ii,jj, 'y')
                    v_z1, v_z2 = Core.coordinatesFD(vv,ii,jj, 'z')
                    w_y0, w_y1, w_y2 = Core.coordinatesCD(ww,ii,jj, 'y')
                    w_z1, w_z2 = Core.coordinatesFD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
                    uy_y = Core.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
                    uz_y = Core.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)

                    ux_z = Core.partialderivativeFD(u_z1, u_z2, z1, z2)
                    uy_z = Core.partialderivativeFD(v_z1, v_z2, z1, z2)
                    uz_z = Core.partialderivativeFD(w_z1, w_z2, z1, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Lower boundary --> z = grid_size-1
                if ((ii > 0 and ii < DIM_Y-2) and (jj == DIM_Z-1)):
                    y0 = yy[ii-1,jj]
                    y2 = yy[ii+1,jj]

                    z0 = zz[ii,jj-1]

                    u_y0, u_y1, u_y2 = Core.coordinatesCD(uu,ii,jj, 'y')
                    u_z0, u_z1 = Core.coordinatesBD(uu,ii,jj, 'z')
                    v_y0, v_y1, v_y2 = Core.coordinatesCD(vv,ii,jj, 'y')
                    v_z0, v_z1 = Core.coordinatesBD(vv,ii,jj, 'z')
                    w_y0, w_y1, w_y2 = Core.coordinatesCD(ww,ii,jj, 'y')
                    w_z0, w_z1 = Core.coordinatesBD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
                    uy_y = Core.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
                    uz_y = Core.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)

                    ux_z = Core.partialderivativeBD(u_z0, u_z1, z0, z1)
                    uy_z = Core.partialderivativeBD(v_z0, v_z1, z0, z1)
                    uz_z = Core.partialderivativeBD(w_z0, w_z1, z0, z1)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Left boundary --> y = 0
                if ((ii == 0) and (jj > 0 and jj < DIM_Z-2)):
                    y2 = yy[ii+1,jj]

                    z0 = zz[ii,jj-1]
                    z2 = zz[ii,jj+1]

                    u_y1, u_y2 = Core.coordinatesFD(uu,ii,jj, 'y')
                    u_z0, u_z1, u_z2 = Core.coordinatesCD(uu,ii,jj, 'z')
                    v_y1, v_y2 = Core.coordinatesFD(vv,ii,jj, 'y')
                    v_z0, v_z1, v_z2 = Core.coordinatesCD(vv,ii,jj, 'z')
                    w_y1, w_y2 = Core.coordinatesFD(ww,ii,jj, 'y')
                    w_z0, w_z1, w_z2 = Core.coordinatesCD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeFD(u_y1, u_y2, y1, y2)
                    uy_y = Core.partialderivativeFD(v_y1, v_y2, y1, y2)
                    uz_y = Core.partialderivativeFD(w_y1, w_y2, y1, y2)

                    ux_z = Core.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
                    uy_z = Core.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
                    uz_z = Core.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Right boundary --> y = grid_size - 1
                if ((ii == DIM_Y-1) and (jj > 0 and jj < DIM_Z-2)):
                    y0 = yy[ii-1,jj]

                    z0 = zz[ii,jj-1]
                    z2 = zz[ii,jj+1]

                    u_y0, u_y1 = Core.coordinatesBD(uu,ii,jj, 'y')
                    u_z0, u_z1, u_z2 = Core.coordinatesCD(uu,ii,jj, 'z')
                    v_y0, v_y1 = Core.coordinatesBD(vv,ii,jj, 'y')
                    v_z0, v_z1, v_z2 = Core.coordinatesCD(vv,ii,jj, 'z')
                    w_y0, w_y1 = Core.coordinatesBD(ww,ii,jj, 'y')
                    w_z0, w_z1, w_z2 = Core.coordinatesCD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeBD(u_y0, u_y1, y0, y1)
                    uy_y = Core.partialderivativeBD(v_y0, v_y1, y0, y1)
                    uz_y = Core.partialderivativeBD(w_y0, w_y1, y0, y1)

                    ux_z = Core.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
                    uy_z = Core.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
                    uz_z = Core.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Left-top corner --> y = 0, z = 0
                if ((ii == 0) and (jj == 0)):
                    y2 = yy[ii+2,jj]

                    z2 = zz[ii,jj+1]

                    u_y1, u_y2 = Core.coordinatesFD(uu,ii,jj, 'y')
                    u_z1, u_z2 = Core.coordinatesFD(uu,ii,jj, 'z')
                    v_y1, v_y2 = Core.coordinatesFD(vv,ii,jj, 'y')
                    v_z1, v_z2 = Core.coordinatesFD(vv,ii,jj, 'z')
                    w_y1, w_y2 = Core.coordinatesFD(ww,ii,jj, 'y')
                    w_z1, w_z2 = Core.coordinatesFD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeFD(u_y1, u_y2, y1, y2)
                    uy_y = Core.partialderivativeFD(v_y1, v_y2, y1, y2)
                    uz_y = Core.partialderivativeFD(w_y1, w_y2, y1, y2)

                    ux_z = Core.partialderivativeFD(u_z1, u_z2, z1, z2)
                    uy_z = Core.partialderivativeFD(v_z1, v_z2, z1, z2)
                    uz_z = Core.partialderivativeFD(w_z1, w_z2, z1, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Right-top corner --> y = grid_size-1, z = 0
                if ((ii == DIM_Y-1) and (jj == 0)):
                    y0 = yy[ii-1,jj]

                    z2 = zz[ii,jj+1]

                    u_y0, u_y1 = Core.coordinatesBD(uu,ii,jj, 'y')
                    u_z1, u_z2 = Core.coordinatesFD(uu,ii,jj, 'z')
                    v_y0, v_y1 = Core.coordinatesBD(vv,ii,jj, 'y')
                    v_z1, v_z2 = Core.coordinatesFD(vv,ii,jj, 'z')
                    w_y0, w_y1 = Core.coordinatesBD(ww,ii,jj, 'y')
                    w_z1, w_z2 = Core.coordinatesFD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeBD(u_y0, u_y1, y0, y1)
                    uy_y = Core.partialderivativeBD(v_y0, v_y1, y0, y1)
                    uz_y = Core.partialderivativeBD(w_y0, w_y1, y0, y1)

                    ux_z = Core.partialderivativeFD(u_z1, u_z2, z1, z2)
                    uy_z = Core.partialderivativeFD(v_z1, v_z2, z1, z2)
                    uz_z = Core.partialderivativeFD(w_z1, w_z2, z1, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Right-bottom corner --> y = grid_size-1, z = grid_size-1
                if ((ii == DIM_Y-1) and (jj == DIM_Z-1)):
                    y0 = yy[ii-1,jj]

                    z0 = zz[ii,jj-1]

                    u_y0, u_y1 = Core.coordinatesBD(uu,ii,jj, 'y')
                    u_z0, u_z1 = Core.coordinatesBD(uu,ii,jj, 'z')
                    v_y0, v_y1 = Core.coordinatesBD(vv,ii,jj, 'y')
                    v_z0, v_z1 = Core.coordinatesBD(vv,ii,jj, 'z')
                    w_y0, w_y1 = Core.coordinatesBD(ww,ii,jj, 'y')
                    w_z0, w_z1 = Core.coordinatesBD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeBD(u_y0, u_y1, y0, y1)
                    uy_y = Core.partialderivativeBD(v_y0, v_y1, y0, y1)
                    uz_y = Core.partialderivativeBD(w_y0, w_y1, y0, y1)

                    ux_z = Core.partialderivativeBD(u_z0, u_z1, z0, z1)
                    uy_z = Core.partialderivativeBD(v_z0, v_z1, z0, z1)
                    uz_z = Core.partialderivativeBD(w_z0, w_z1, z0, z1)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Left-bottom corner --> y = 0, z = grid_size-1
                if ((ii == 0) and (jj == DIM_Z-1)):
                    y2 = yy[ii+1,jj]

                    z0 = zz[ii,jj-1]

                    u_y1, u_y2 = Core.coordinatesFD(uu,ii,jj, 'y')
                    u_z0, u_z1 = Core.coordinatesBD(uu,ii,jj, 'z')
                    v_y1, v_y2 = Core.coordinatesFD(vv,ii,jj, 'y')
                    v_z0, v_z1 = Core.coordinatesBD(vv,ii,jj, 'z')
                    w_y1, w_y2 = Core.coordinatesFD(ww,ii,jj, 'y')
                    w_z0, w_z1 = Core.coordinatesBD(ww,ii,jj, 'z')

                    ux_y = Core.partialderivativeFD(u_y1, u_y2, y1, y2)
                    uy_y = Core.partialderivativeFD(v_y1, v_y2, y1, y2)
                    uz_y = Core.partialderivativeFD(w_y1, w_y2, y1, y2)

                    ux_z = Core.partialderivativeBD(u_z0, u_z1, z0, z1)
                    uy_z = Core.partialderivativeBD(v_z0, v_z1, z0, z1)
                    uz_z = Core.partialderivativeBD(w_z0, w_z1, z0, z1)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

        return gradient

# import numpy as np
# import csv
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import math as m
# from mpl_toolkits.mplot3d import Axes3D
# from collections import defaultdict
#
# class Core:
#
#     object_counter = 0
#
#     def __init__(self):
#         self.object_id = Core.object_counter
#         Core.object_counter += 1
#
#     def importCoordinates(coordinate, usecase): ##### Input coordinate should be 'y' or 'z'
#         ## As for RA=1 the grid is square, there is no file containing y coordinates for RA.
#         ## In this case, y and z coordinates are identical.
#         RA = int(usecase.split('_')[0])
#
#         if (coordinate == 'y'):
#             if (RA == 1): ## In case of a square grid, y and z are identical
#                 coordinate = 'z'
#
#         ## Open appropriate file and read it out
#         dataline = []
#         filepath = '../inversion/DATA/RECTANGULARDUCT/DATA/'+coordinate+'coord_' + usecase + '.prof.txt'
#         data = []
#         with open(filepath, 'r', encoding='latin-1') as fd:
#             for line in fd:
#                 dataline = line.split()
#                 if len(dataline) != 0:
#                     data = dataline
#
#             DIM = len(data)
#
#         fd.close()
#
#         coord = np.zeros(DIM)
#         for ii in range(DIM-1):
#             coord[ii] = data[ii]
#         return coord, DIM
#
#     def importMeanVelocity(self, DIM_Y, DIM_Z, usecase, velocity_component):
#         filepath = '../inversion/DATA/RECTANGULARDUCT/DATA/'+str(velocity_component)+'_' + str(usecase) + '.prof.txt'
#         U = np.zeros([DIM_Y, DIM_Z])
#
#         counter = 0
#         with open(filepath, encoding='latin-1') as fd:
#             for i in range(24):
#                 next(fd)
#             for line in fd:
#                 dataline= line.split()
#                 if len(dataline) == DIM_Z: ## Omit blank lines
#                     U[counter,:] = dataline
#                     counter += 1
#
#         fd.close()
#
#         return U
#
#     def importStressTensor(self, usecase, DIM_Y, DIM_Z):
#         ## Open appropriate file and read it out
#
#         components = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
#         data = defaultdict(list)
#         for ii in range(len(components)):
#             comp = components[ii]
#             filepath = '../inversion/DATA/RECTANGULARDUCT/DATA/'+comp+ '_' + usecase + '.prof.txt'
#             data[comp]  = []
#             with open(filepath, 'r', encoding='latin-1') as fd:
#                 for line in fd:
#                     if line[0] != '%':
#                         data[comp].append(line.split())
#                         break
#
#                 for line in fd:
#                     if len(line.split()) != 0: ## Omit blank lines
#                         data[comp].append(line.split())
#
#             fd.close()
#
#         tensor = np.ones([DIM_Y, DIM_Z, 3,3])
#         counter = 1
#         for ii in range(DIM_Y):
#             for jj in range(DIM_Z):
#                 tensor[ii,:,0,0] = data['uu'][ii]
#                 tensor[ii,:,1,0] = data['uv'][ii]
#                 tensor[ii,:,2,0] = data['uw'][ii]
#
#                 tensor[ii,:,0,1] = data['uv'][ii]
#                 tensor[ii,:,1,1] = data['vv'][ii]
#                 tensor[ii,:,2,1] = data['vw'][ii]
#
#                 tensor[ii,:,0,2] = data['uw'][ii]
#                 tensor[ii,:,1,2] = data['vw'][ii]
#                 tensor[ii,:,2,2] = data['ww'][ii]
#
#                 counter += 1
#
#         return tensor
#
#     def plotMeanVelocityComponent(self, RA, Retau, velocity_component):
#
#         fig = plt.figure(1,figsize=(10,30))
#         gs = fig.add_gridspec(6, 14)
#         fig.suptitle('Mean velocity flow for $R_{\\tau}$ and different RA values for velocity compoment '+ velocity_component)
#         fig.subplots_adjust(hspace = 1)
#         for ii in range(len(RA)):
#             ra = RA[ii]
#             ncols = RA[ii]-1
#
#             ### Collect coordinates and mean velocity data
#             usecase = str(ra)+'_'+str(Retau)
#
#             zcoord, DIM_Z = Core.importCoordinates('z', usecase)
#             ycoord, DIM_Y = Core.importCoordinates('y', usecase)
#
#             U = Core.importMeanVelocity(self, DIM_Y, DIM_Z, usecase, velocity_component)
#
#             ### Create subplots
#             if (ncols == 0):
#                 ax = fig.add_subplot(gs[ii, 0])
#             else:
#                 ax = fig.add_subplot(gs[ii, 0:ncols])
#
#             ### Add contour graphs to subplots
#             ax.contourf(U, extent=(np.amin(zcoord), np.amax(zcoord), np.amin(ycoord), np.amax(ycoord)), cmap=plt.cm.Oranges)
#             ax.set_title('RA = '+str(ra))
#
#             if (ii == m.ceil(len(RA)/2)):
#                 ax.set_ylabel('z: spanwise (a. u.)')
#
#             if (ii == len(RA)-1):
#                 ax.set_xlabel('y: wall normal (a. u. )')
#
#         plt.show()
#
#
#     def plotMeanVelocityField(self, RA, Retau, data, ycoord, zcoord):
#         x, y, z = np.meshgrid(ycoord[1:-1:10], zcoord[1:-1:10], np.zeros(1))
#
#         fig_field = plt.figure(2)
#         ax = fig_field.gca(projection='3d')
#         ax.quiver(x, y, z, data[1:-1:10,1:-1:10,0], data[1:-1:10,1:-1:10,1], data[1:-1:10,1:-1:10,2], length=0.05, normalize=True)
#         ax.set_title('Mean Velocity field for $R_{\\tau}$ = '+str(Retau) +' and RA = '+str(RA))
#         plt.show()
#
#
#     def calc_S_R_test(self, grad_u, k, eps):
#         n = grad_u.shape[0]
#         S = np.zeros((n, 3, 3))
#         R = np.zeros((n, 3, 3))
#         ke = k / eps
#         for i in range(n):
#             S[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
#             R[i, :, :] = ke[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))
#
#         return S,R
#
#     def calc_T_test(self, S, R):
#
#         num_points = S.shape[0]
#         num_tensor_basis = 10
#         T = np.zeros((num_points, num_tensor_basis, 3, 3))
#         for i in range(num_points):
#             sij = S[i, :, :]
#             rij = R[i, :, :]
#             T[i, 0, :, :] = sij
#             T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
#             T[i, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
#             T[i, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
#             T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
#             T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) + np.dot(sij, np.dot(rij, rij)) - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
#             T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
#             T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
#             T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) + np.dot(np.dot(sij, sij), np.dot(rij, rij))- 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
#             T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
#             T_flat = np.zeros((num_points, num_tensor_basis, 9))
#         for i in range(3):
#             for j in range(3):
#                 T_flat[:, :, 3*i+j] = T[:, :, i, j]
#         return T_flat
#
#     def calc_scalar_basis_test(self, S, R):
#         num_points = S.shape[0]
#         num_eigenvalues = 5
#         eigenvalues = np.zeros((num_points, num_eigenvalues))
#         for i in range(num_points):
#             eigenvalues[i, 0] = np.trace(np.dot(S[i, :, :], S[i, :, :]))
#             eigenvalues[i, 1] = np.trace(np.dot(R[i, :, :], R[i, :, :]))
#             eigenvalues[i, 2] = np.trace(np.dot(S[i, :, :], np.dot(S[i, :, :], S[i, :, :])))
#             eigenvalues[i, 3] = np.trace(np.dot(R[i, :, :], np.dot(R[i, :, :], S[i, :, :])))
#             eigenvalues[i, 4] = np.trace(np.dot(np.dot(R[i, :, :], R[i, :, :]), np.dot(S[i, :, :], S[i, :, :])))
#
#         return eigenvalues
#
#
#     def calc_S_R(self, grad_u, k, eps):
#         DIM_Y = grad_u.shape[0]
#         DIM_Z = grad_u.shape[1]
#
#         S = np.zeros((DIM_Y, DIM_Z, 3, 3))
#         R = np.zeros((DIM_Y, DIM_Z, 3, 3))
#         ke = k / eps
#
#         for ii in range(DIM_Y):
#             for jj in range(DIM_Z):
#                 S[ii, jj, :, :] = ke[ii, jj] * 0.5 * (grad_u[ii, jj, :, :] + np.transpose(grad_u[ii, jj,  :, :]))
#                 R[ii, jj, :, :] = ke[ii, jj] * 0.5 * (grad_u[ii, jj, :, :] - np.transpose(grad_u[ii, jj, :,  :]))
#
#         return S,R
#
#     def calc_tensor_basis(self, S, R):
#         DIM_Y = S.shape[0]
#         DIM_Z = S.shape[1]
#
#         T = np.zeros((DIM_Y, DIM_Z, 10, 3, 3))
#         for ii in range(DIM_Y):
#             for jj in range(DIM_Z):
#                 sij = S[ii, jj, :, :]
#                 rij = R[ii, jj, :, :]
#                 T[ii, jj, 0, :, :] = sij
#                 T[ii, jj, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
#                 T[ii, jj, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
#                 T[ii, jj, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
#                 T[ii, jj, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
#                 T[ii, jj, 5, :, :] = np.dot(rij, np.dot(rij, sij)) + np.dot(sij, np.dot(rij, rij)) - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
#                 T[ii, jj, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
#                 T[ii, jj, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
#                 T[ii, jj, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) + np.dot(np.dot(sij, sij), np.dot(rij, rij))- 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
#                 T[ii, jj, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
#
#         T_flat = np.zeros((num_points, num_tensor_basis, 9))
#         for i in range(3):
#             for j in range(3):
#                 T_flat[:, :, 3*i+j] = T[:, :, i, j]
#         return T_flat
#
#     def calc_scalar_basis(self, S, R):
#         DIM_Y = R.shape[0]
#         DIM_Z = R.shape[1]
#
#         eigenvalues = np.zeros([DIM_Y, DIM_Z, 5])
#         for ii in range(DIM_Y):
#             for jj in range(DIM_Z):
#                 eigenvalues[ii, jj, 0] = np.trace(np.dot(S[ii, jj, :, :], S[ii, jj, :, :]))
#                 eigenvalues[ii, jj, 1] = np.trace(np.dot(R[ii, jj, :, :], R[ii, jj, :, :]))
#                 eigenvalues[ii, jj, 2] = np.trace(np.dot(S[ii, jj, :, :], np.dot(S[ii, jj, :, :], S[ii, jj, :, :])))
#                 eigenvalues[ii, jj, 3] = np.trace(np.dot(R[ii, jj, :, :], np.dot(R[ii, jj, :, :], S[ii, jj, :, :])))
#                 eigenvalues[ii, jj, 4] = np.trace(np.dot(np.dot(R[ii, jj, :, :], R[ii, jj, :, :]), np.dot(S[ii, jj, :, :], S[ii, jj, :, :])))
#
#         return eigenvalues
#
#     def calc_output(self, tau, k):
#
#         num_points = tau.shape[0]
#         b = np.zeros((num_points, 3, 3))
#
#         for i in range(3):
#             for j in range(3):
#                 b[:, i, j] = tau[:, i, j]/(2.0 * k)
#             b[:, i, i] -= 1./3.
#
#         b_vector = np.zeros((num_points, 9))
#         for i in range(3):
#             for j in range(3):
#                 b_vector[:, 3*i+j] = b[:, i, j]
#         return b_vector
#
#
#     def calc_k(self, velocity_gradient):
#         DIM_Y = np.shape(velocity_gradient)[0]
#         DIM_Z = np.shape(velocity_gradient)[1]
#
#         k = np.zeros([DIM_Y, DIM_Z])
#
#         ## Implementation of formula: k = 0.5*Tr(tau)
#         ## Tr(tau) = (sum of diagonal elements of tau)
#
#         for ii in range(DIM_Y):
#             for jj in range(DIM_Z):
#                 for cc in range(3):
#                     k[ii,jj] += velocity_gradient[ii,jj,cc,cc]
#
#         k = 0.5*k
#         return k
#
#     def partialderivativeCD(u0, u1, u2, q0, q2):
#         return (u0-2*u1+u2)/(q2-q0)
#
#     def partialderivativeFD(u1, u2, q1, q2):
#         return (u2-u1)/(q2-q1)
#
#     def partialderivativeBD(u0, u1, q0, q1):
#         return (u1-u0)/(q1-q0)
#
#     def coordinatesCD(uu,ii,jj, direction):
#         if direction == 'y':
#             return uu[ii-1,jj], uu[ii,jj], uu[ii+1,jj]
#         else:
#             return uu[ii,jj-1], uu[ii,jj], uu[ii,jj+1]
#
#     def coordinatesFD(uu,ii,jj, direction):
#         if direction == 'y':
#             return uu[ii,jj], uu[ii+1,jj]
#         else:
#             return uu[ii,jj], uu[ii,jj+1]
#
#     def coordinatesBD(uu,ii,jj, direction):
#         if direction == 'y':
#             return uu[ii-1,jj], uu[ii,jj]
#         else:
#             return uu[ii,jj-1], uu[ii,jj]
#
#     def gradient(self,data,ycoord, zcoord):
#
#         DIM_Y = len(ycoord)
#         DIM_Z = len(zcoord)
#
#
#         zz, yy = np.meshgrid(zcoord, ycoord)
#
#         uu = data[:,:,0]
#         vv = data[:,:,1]
#         ww = data[:,:,2]
#
#         ## Create matrix containing u,v and w for every (x,y) data point
#         ## Matrix[X, Y, 3 (u,v,w)]
#         u = np.array([uu,vv,ww])
#
#         ## Calculate gradient using numpy-function. Returns three lists,
#         ## each containing an array corresponding to the derivative to one dimension.
#
#         gradient = np.zeros([DIM_Y,DIM_Z,3,3])
#
#         ## As the flow is symmetric in the x-direction, the gradient in x-direction is always zero.
#         ux_x = 0
#         uy_x = 0
#         uz_x = 0
#
#         for ii in range(DIM_Y):
#             for jj in range(DIM_Z):
#                 y1 = yy[ii,jj]
#                 z1 = zz[ii,jj]
#
#                 ## Inside grid points
#                 if ((ii > 0 and ii < DIM_Y-2) and (jj > 0 and jj < DIM_Z-2)):
#                     y0 = yy[ii-1,jj]
#                     y2 = yy[ii+1,jj]
#
#                     z0 = zz[ii,jj-1]
#                     z2 = zz[ii,jj+1]
#
#                     u_y0, u_y1, u_y2 = Core.coordinatesCD(uu,ii,jj, 'y')
#                     u_z0, u_z1, u_z2 = Core.coordinatesCD(uu,ii,jj, 'z')
#                     v_y0, v_y1, v_y2 = Core.coordinatesCD(vv,ii,jj, 'y')
#                     v_z0, v_z1, v_z2 = Core.coordinatesCD(vv,ii,jj, 'z')
#                     w_y0, w_y1, w_y2 = Core.coordinatesCD(ww,ii,jj, 'y')
#                     w_z0, w_z1, w_z2 = Core.coordinatesCD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
#                     uy_y = Core.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
#                     uz_y = Core.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)
#
#                     ux_z = Core.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
#                     uy_z = Core.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
#                     uz_z = Core.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Upper boundary --> z = 0
#                 if ((ii > 0 and ii < DIM_Y-2) and (jj == 0)):
#                     y0 = yy[ii-1,jj]
#                     y2 = yy[ii+1,jj]
#
#                     z2 = zz[ii,jj+1]
#
#                     u_y0, u_y1, u_y2 = Core.coordinatesCD(uu,ii,jj, 'y')
#                     u_z1, u_z2 = Core.coordinatesFD(uu,ii,jj, 'z')
#                     v_y0, v_y1, v_y2 = Core.coordinatesCD(vv,ii,jj, 'y')
#                     v_z1, v_z2 = Core.coordinatesFD(vv,ii,jj, 'z')
#                     w_y0, w_y1, w_y2 = Core.coordinatesCD(ww,ii,jj, 'y')
#                     w_z1, w_z2 = Core.coordinatesFD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
#                     uy_y = Core.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
#                     uz_y = Core.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)
#
#                     ux_z = Core.partialderivativeFD(u_z1, u_z2, z1, z2)
#                     uy_z = Core.partialderivativeFD(v_z1, v_z2, z1, z2)
#                     uz_z = Core.partialderivativeFD(w_z1, w_z2, z1, z2)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Lower boundary --> z = grid_size-1
#                 if ((ii > 0 and ii < DIM_Y-2) and (jj == DIM_Z-1)):
#                     y0 = yy[ii-1,jj]
#                     y2 = yy[ii+1,jj]
#
#                     z0 = zz[ii,jj-1]
#
#                     u_y0, u_y1, u_y2 = Core.coordinatesCD(uu,ii,jj, 'y')
#                     u_z0, u_z1 = Core.coordinatesBD(uu,ii,jj, 'z')
#                     v_y0, v_y1, v_y2 = Core.coordinatesCD(vv,ii,jj, 'y')
#                     v_z0, v_z1 = Core.coordinatesBD(vv,ii,jj, 'z')
#                     w_y0, w_y1, w_y2 = Core.coordinatesCD(ww,ii,jj, 'y')
#                     w_z0, w_z1 = Core.coordinatesBD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
#                     uy_y = Core.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
#                     uz_y = Core.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)
#
#                     ux_z = Core.partialderivativeBD(u_z0, u_z1, z0, z1)
#                     uy_z = Core.partialderivativeBD(v_z0, v_z1, z0, z1)
#                     uz_z = Core.partialderivativeBD(w_z0, w_z1, z0, z1)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Left boundary --> y = 0
#                 if ((ii == 0) and (jj > 0 and jj < DIM_Z-2)):
#                     y2 = yy[ii+1,jj]
#
#                     z0 = zz[ii,jj-1]
#                     z2 = zz[ii,jj+1]
#
#                     u_y1, u_y2 = Core.coordinatesFD(uu,ii,jj, 'y')
#                     u_z0, u_z1, u_z2 = Core.coordinatesCD(uu,ii,jj, 'z')
#                     v_y1, v_y2 = Core.coordinatesFD(vv,ii,jj, 'y')
#                     v_z0, v_z1, v_z2 = Core.coordinatesCD(vv,ii,jj, 'z')
#                     w_y1, w_y2 = Core.coordinatesFD(ww,ii,jj, 'y')
#                     w_z0, w_z1, w_z2 = Core.coordinatesCD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeFD(u_y1, u_y2, y1, y2)
#                     uy_y = Core.partialderivativeFD(v_y1, v_y2, y1, y2)
#                     uz_y = Core.partialderivativeFD(w_y1, w_y2, y1, y2)
#
#                     ux_z = Core.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
#                     uy_z = Core.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
#                     uz_z = Core.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Right boundary --> y = grid_size - 1
#                 if ((ii == DIM_Y-1) and (jj > 0 and jj < DIM_Z-2)):
#                     y0 = yy[ii-1,jj]
#
#                     z0 = zz[ii,jj-1]
#                     z2 = zz[ii,jj+1]
#
#                     u_y0, u_y1 = Core.coordinatesBD(uu,ii,jj, 'y')
#                     u_z0, u_z1, u_z2 = Core.coordinatesCD(uu,ii,jj, 'z')
#                     v_y0, v_y1 = Core.coordinatesBD(vv,ii,jj, 'y')
#                     v_z0, v_z1, v_z2 = Core.coordinatesCD(vv,ii,jj, 'z')
#                     w_y0, w_y1 = Core.coordinatesBD(ww,ii,jj, 'y')
#                     w_z0, w_z1, w_z2 = Core.coordinatesCD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeBD(u_y0, u_y1, y0, y1)
#                     uy_y = Core.partialderivativeBD(v_y0, v_y1, y0, y1)
#                     uz_y = Core.partialderivativeBD(w_y0, w_y1, y0, y1)
#
#                     ux_z = Core.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
#                     uy_z = Core.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
#                     uz_z = Core.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Left-top corner --> y = 0, z = 0
#                 if ((ii == 0) and (jj == 0)):
#                     y2 = yy[ii+2,jj]
#
#                     z2 = zz[ii,jj+1]
#
#                     u_y1, u_y2 = Core.coordinatesFD(uu,ii,jj, 'y')
#                     u_z1, u_z2 = Core.coordinatesFD(uu,ii,jj, 'z')
#                     v_y1, v_y2 = Core.coordinatesFD(vv,ii,jj, 'y')
#                     v_z1, v_z2 = Core.coordinatesFD(vv,ii,jj, 'z')
#                     w_y1, w_y2 = Core.coordinatesFD(ww,ii,jj, 'y')
#                     w_z1, w_z2 = Core.coordinatesFD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeFD(u_y1, u_y2, y1, y2)
#                     uy_y = Core.partialderivativeFD(v_y1, v_y2, y1, y2)
#                     uz_y = Core.partialderivativeFD(w_y1, w_y2, y1, y2)
#
#                     ux_z = Core.partialderivativeFD(u_z1, u_z2, z1, z2)
#                     uy_z = Core.partialderivativeFD(v_z1, v_z2, z1, z2)
#                     uz_z = Core.partialderivativeFD(w_z1, w_z2, z1, z2)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Right-top corner --> y = grid_size-1, z = 0
#                 if ((ii == DIM_Y-1) and (jj == 0)):
#                     y0 = yy[ii-1,jj]
#
#                     z2 = zz[ii,jj+1]
#
#                     u_y0, u_y1 = Core.coordinatesBD(uu,ii,jj, 'y')
#                     u_z1, u_z2 = Core.coordinatesFD(uu,ii,jj, 'z')
#                     v_y0, v_y1 = Core.coordinatesBD(vv,ii,jj, 'y')
#                     v_z1, v_z2 = Core.coordinatesFD(vv,ii,jj, 'z')
#                     w_y0, w_y1 = Core.coordinatesBD(ww,ii,jj, 'y')
#                     w_z1, w_z2 = Core.coordinatesFD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeBD(u_y0, u_y1, y0, y1)
#                     uy_y = Core.partialderivativeBD(v_y0, v_y1, y0, y1)
#                     uz_y = Core.partialderivativeBD(w_y0, w_y1, y0, y1)
#
#                     ux_z = Core.partialderivativeFD(u_z1, u_z2, z1, z2)
#                     uy_z = Core.partialderivativeFD(v_z1, v_z2, z1, z2)
#                     uz_z = Core.partialderivativeFD(w_z1, w_z2, z1, z2)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Right-bottom corner --> y = grid_size-1, z = grid_size-1
#                 if ((ii == DIM_Y-1) and (jj == DIM_Z-1)):
#                     y0 = yy[ii-1,jj]
#
#                     z0 = zz[ii,jj-1]
#
#                     u_y0, u_y1 = Core.coordinatesBD(uu,ii,jj, 'y')
#                     u_z0, u_z1 = Core.coordinatesBD(uu,ii,jj, 'z')
#                     v_y0, v_y1 = Core.coordinatesBD(vv,ii,jj, 'y')
#                     v_z0, v_z1 = Core.coordinatesBD(vv,ii,jj, 'z')
#                     w_y0, w_y1 = Core.coordinatesBD(ww,ii,jj, 'y')
#                     w_z0, w_z1 = Core.coordinatesBD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeBD(u_y0, u_y1, y0, y1)
#                     uy_y = Core.partialderivativeBD(v_y0, v_y1, y0, y1)
#                     uz_y = Core.partialderivativeBD(w_y0, w_y1, y0, y1)
#
#                     ux_z = Core.partialderivativeBD(u_z0, u_z1, z0, z1)
#                     uy_z = Core.partialderivativeBD(v_z0, v_z1, z0, z1)
#                     uz_z = Core.partialderivativeBD(w_z0, w_z1, z0, z1)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#                 ## Left-bottom corner --> y = 0, z = grid_size-1
#                 if ((ii == 0) and (jj == DIM_Z-1)):
#                     y2 = yy[ii+1,jj]
#
#                     z0 = zz[ii,jj-1]
#
#                     u_y1, u_y2 = Core.coordinatesFD(uu,ii,jj, 'y')
#                     u_z0, u_z1 = Core.coordinatesBD(uu,ii,jj, 'z')
#                     v_y1, v_y2 = Core.coordinatesFD(vv,ii,jj, 'y')
#                     v_z0, v_z1 = Core.coordinatesBD(vv,ii,jj, 'z')
#                     w_y1, w_y2 = Core.coordinatesFD(ww,ii,jj, 'y')
#                     w_z0, w_z1 = Core.coordinatesBD(ww,ii,jj, 'z')
#
#                     ux_y = Core.partialderivativeFD(u_y1, u_y2, y1, y2)
#                     uy_y = Core.partialderivativeFD(v_y1, v_y2, y1, y2)
#                     uz_y = Core.partialderivativeFD(w_y1, w_y2, y1, y2)
#
#                     ux_z = Core.partialderivativeBD(u_z0, u_z1, z0, z1)
#                     uy_z = Core.partialderivativeBD(v_z0, v_z1, z0, z1)
#                     uz_z = Core.partialderivativeBD(w_z0, w_z1, z0, z1)
#
#                     gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
#
#         return gradient
#
#
#     def load_test_data():
#         data = np.loadtxt("nn/test_data.txt", skiprows=4)
#         k = data[:, 0]
#         eps = data[:, 1]
#         grad_u_flat = data[:, 2:11]
#         stresses_flat = data[:, 11:]
#
#         num_points = data.shape[0]
#         grad_u = np.zeros((num_points, 3, 3))
#         stresses = np.zeros((num_points, 3, 3))
#         for i in range(3):
#             for j in range(3):
#                 grad_u[:, i, j] = grad_u_flat[:, i*3+j]
#                 stresses[:, i, j] = stresses_flat[:, i*3+j]
#
#         return k, eps, grad_u, stresses
#
#     def plot_results(predicted_stresses, true_stresses):
#
#         fig = plt.figure()
#         fig.patch.set_facecolor('white')
#         on_diag = [0, 4, 8]
#         for i in range(9):
#                 plt.subplot(3, 3, i+1)
#                 ax = fig.gca()
#                 ax.set_aspect('equal')
#                 plt.plot([-1., 1.], [-1., 1.], 'r--')
#                 plt.scatter(true_stresses[:, i], predicted_stresses[:, i])
#                 plt.xlabel('True value')
#                 plt.ylabel('Predicted value')
#                 idx_1 = i / 3
#                 idx_2 = i % 3
#                 plt.title('A' + str(idx_1) + str(idx_2))
#                 if i in on_diag:
#                     plt.xlim([-1./3., 2./3.])
#                     plt.ylim([-1./3., 2./3.])
#                 else:
#                     plt.xlim([-0.5, 0.5])
#                     plt.ylim([-0.5, 0.5])
#         plt.tight_layout()
