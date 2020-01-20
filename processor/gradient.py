import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m
from collections import defaultdict

class Gradient:

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

    def gradient(data,ycoord, zcoord):

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

                    u_y0, u_y1, u_y2 = Gradient.coordinatesCD(uu,ii,jj, 'y')
                    u_z0, u_z1, u_z2 = Gradient.coordinatesCD(uu,ii,jj, 'z')
                    v_y0, v_y1, v_y2 = Gradient.coordinatesCD(vv,ii,jj, 'y')
                    v_z0, v_z1, v_z2 = Gradient.coordinatesCD(vv,ii,jj, 'z')
                    w_y0, w_y1, w_y2 = Gradient.coordinatesCD(ww,ii,jj, 'y')
                    w_z0, w_z1, w_z2 = Gradient.coordinatesCD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
                    uy_y = Gradient.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
                    uz_y = Gradient.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)

                    ux_z = Gradient.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
                    uy_z = Gradient.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
                    uz_z = Gradient.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Upper boundary --> z = 0
                if ((ii > 0 and ii < DIM_Y-2) and (jj == 0)):
                    y0 = yy[ii-1,jj]
                    y2 = yy[ii+1,jj]

                    z2 = zz[ii,jj+1]

                    u_y0, u_y1, u_y2 = Gradient.coordinatesCD(uu,ii,jj, 'y')
                    u_z1, u_z2 = Gradient.coordinatesFD(uu,ii,jj, 'z')
                    v_y0, v_y1, v_y2 = Gradient.coordinatesCD(vv,ii,jj, 'y')
                    v_z1, v_z2 = Gradient.coordinatesFD(vv,ii,jj, 'z')
                    w_y0, w_y1, w_y2 = Gradient.coordinatesCD(ww,ii,jj, 'y')
                    w_z1, w_z2 = Gradient.coordinatesFD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
                    uy_y = Gradient.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
                    uz_y = Gradient.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)

                    ux_z = Gradient.partialderivativeFD(u_z1, u_z2, z1, z2)
                    uy_z = Gradient.partialderivativeFD(v_z1, v_z2, z1, z2)
                    uz_z = Gradient.partialderivativeFD(w_z1, w_z2, z1, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Lower boundary --> z = grid_size-1
                if ((ii > 0 and ii < DIM_Y-2) and (jj == DIM_Z-1)):
                    y0 = yy[ii-1,jj]
                    y2 = yy[ii+1,jj]

                    z0 = zz[ii,jj-1]

                    u_y0, u_y1, u_y2 = Gradient.coordinatesCD(uu,ii,jj, 'y')
                    u_z0, u_z1 = Gradient.coordinatesBD(uu,ii,jj, 'z')
                    v_y0, v_y1, v_y2 = Gradient.coordinatesCD(vv,ii,jj, 'y')
                    v_z0, v_z1 = Gradient.coordinatesBD(vv,ii,jj, 'z')
                    w_y0, w_y1, w_y2 = Gradient.coordinatesCD(ww,ii,jj, 'y')
                    w_z0, w_z1 = Gradient.coordinatesBD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeCD(u_y0, u_y1, u_y2, y0, y2)
                    uy_y = Gradient.partialderivativeCD(v_y0, v_y1, v_y2, y0, y2)
                    uz_y = Gradient.partialderivativeCD(w_y0, w_y1, w_y2, y0, y2)

                    ux_z = Gradient.partialderivativeBD(u_z0, u_z1, z0, z1)
                    uy_z = Gradient.partialderivativeBD(v_z0, v_z1, z0, z1)
                    uz_z = Gradient.partialderivativeBD(w_z0, w_z1, z0, z1)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Left boundary --> y = 0
                if ((ii == 0) and (jj > 0 and jj < DIM_Z-2)):
                    y2 = yy[ii+1,jj]

                    z0 = zz[ii,jj-1]
                    z2 = zz[ii,jj+1]

                    u_y1, u_y2 = Gradient.coordinatesFD(uu,ii,jj, 'y')
                    u_z0, u_z1, u_z2 = Gradient.coordinatesCD(uu,ii,jj, 'z')
                    v_y1, v_y2 = Gradient.coordinatesFD(vv,ii,jj, 'y')
                    v_z0, v_z1, v_z2 = Gradient.coordinatesCD(vv,ii,jj, 'z')
                    w_y1, w_y2 = Gradient.coordinatesFD(ww,ii,jj, 'y')
                    w_z0, w_z1, w_z2 = Gradient.coordinatesCD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeFD(u_y1, u_y2, y1, y2)
                    uy_y = Gradient.partialderivativeFD(v_y1, v_y2, y1, y2)
                    uz_y = Gradient.partialderivativeFD(w_y1, w_y2, y1, y2)

                    ux_z = Gradient.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
                    uy_z = Gradient.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
                    uz_z = Gradient.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Right boundary --> y = grid_size - 1
                if ((ii == DIM_Y-1) and (jj > 0 and jj < DIM_Z-2)):
                    y0 = yy[ii-1,jj]

                    z0 = zz[ii,jj-1]
                    z2 = zz[ii,jj+1]

                    u_y0, u_y1 = Gradient.coordinatesBD(uu,ii,jj, 'y')
                    u_z0, u_z1, u_z2 = Gradient.coordinatesCD(uu,ii,jj, 'z')
                    v_y0, v_y1 = Gradient.coordinatesBD(vv,ii,jj, 'y')
                    v_z0, v_z1, v_z2 = Gradient.coordinatesCD(vv,ii,jj, 'z')
                    w_y0, w_y1 = Gradient.coordinatesBD(ww,ii,jj, 'y')
                    w_z0, w_z1, w_z2 = Gradient.coordinatesCD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeBD(u_y0, u_y1, y0, y1)
                    uy_y = Gradient.partialderivativeBD(v_y0, v_y1, y0, y1)
                    uz_y = Gradient.partialderivativeBD(w_y0, w_y1, y0, y1)

                    ux_z = Gradient.partialderivativeCD(u_z0, u_z1, u_z2, z0, z2)
                    uy_z = Gradient.partialderivativeCD(v_z0, v_z1, v_z2, z0, z2)
                    uz_z = Gradient.partialderivativeCD(w_z0, w_z1, w_z2, z0, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Left-top corner --> y = 0, z = 0
                if ((ii == 0) and (jj == 0)):
                    y2 = yy[ii+2,jj]

                    z2 = zz[ii,jj+1]

                    u_y1, u_y2 = Gradient.coordinatesFD(uu,ii,jj, 'y')
                    u_z1, u_z2 = Gradient.coordinatesFD(uu,ii,jj, 'z')
                    v_y1, v_y2 = Gradient.coordinatesFD(vv,ii,jj, 'y')
                    v_z1, v_z2 = Gradient.coordinatesFD(vv,ii,jj, 'z')
                    w_y1, w_y2 = Gradient.coordinatesFD(ww,ii,jj, 'y')
                    w_z1, w_z2 = Gradient.coordinatesFD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeFD(u_y1, u_y2, y1, y2)
                    uy_y = Gradient.partialderivativeFD(v_y1, v_y2, y1, y2)
                    uz_y = Gradient.partialderivativeFD(w_y1, w_y2, y1, y2)

                    ux_z = Gradient.partialderivativeFD(u_z1, u_z2, z1, z2)
                    uy_z = Gradient.partialderivativeFD(v_z1, v_z2, z1, z2)
                    uz_z = Gradient.partialderivativeFD(w_z1, w_z2, z1, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Right-top corner --> y = grid_size-1, z = 0
                if ((ii == DIM_Y-1) and (jj == 0)):
                    y0 = yy[ii-1,jj]

                    z2 = zz[ii,jj+1]

                    u_y0, u_y1 = Gradient.coordinatesBD(uu,ii,jj, 'y')
                    u_z1, u_z2 = Gradient.coordinatesFD(uu,ii,jj, 'z')
                    v_y0, v_y1 = Gradient.coordinatesBD(vv,ii,jj, 'y')
                    v_z1, v_z2 = Gradient.coordinatesFD(vv,ii,jj, 'z')
                    w_y0, w_y1 = Gradient.coordinatesBD(ww,ii,jj, 'y')
                    w_z1, w_z2 = Gradient.coordinatesFD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeBD(u_y0, u_y1, y0, y1)
                    uy_y = Gradient.partialderivativeBD(v_y0, v_y1, y0, y1)
                    uz_y = Gradient.partialderivativeBD(w_y0, w_y1, y0, y1)

                    ux_z = Gradient.partialderivativeFD(u_z1, u_z2, z1, z2)
                    uy_z = Gradient.partialderivativeFD(v_z1, v_z2, z1, z2)
                    uz_z = Gradient.partialderivativeFD(w_z1, w_z2, z1, z2)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Right-bottom corner --> y = grid_size-1, z = grid_size-1
                if ((ii == DIM_Y-1) and (jj == DIM_Z-1)):
                    y0 = yy[ii-1,jj]

                    z0 = zz[ii,jj-1]

                    u_y0, u_y1 = Gradient.coordinatesBD(uu,ii,jj, 'y')
                    u_z0, u_z1 = Gradient.coordinatesBD(uu,ii,jj, 'z')
                    v_y0, v_y1 = Gradient.coordinatesBD(vv,ii,jj, 'y')
                    v_z0, v_z1 = Gradient.coordinatesBD(vv,ii,jj, 'z')
                    w_y0, w_y1 = Gradient.coordinatesBD(ww,ii,jj, 'y')
                    w_z0, w_z1 = Gradient.coordinatesBD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeBD(u_y0, u_y1, y0, y1)
                    uy_y = Gradient.partialderivativeBD(v_y0, v_y1, y0, y1)
                    uz_y = Gradient.partialderivativeBD(w_y0, w_y1, y0, y1)

                    ux_z = Gradient.partialderivativeBD(u_z0, u_z1, z0, z1)
                    uy_z = Gradient.partialderivativeBD(v_z0, v_z1, z0, z1)
                    uz_z = Gradient.partialderivativeBD(w_z0, w_z1, z0, z1)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

                ## Left-bottom corner --> y = 0, z = grid_size-1
                if ((ii == 0) and (jj == DIM_Z-1)):
                    y2 = yy[ii+1,jj]

                    z0 = zz[ii,jj-1]

                    u_y1, u_y2 = Gradient.coordinatesFD(uu,ii,jj, 'y')
                    u_z0, u_z1 = Gradient.coordinatesBD(uu,ii,jj, 'z')
                    v_y1, v_y2 = Gradient.coordinatesFD(vv,ii,jj, 'y')
                    v_z0, v_z1 = Gradient.coordinatesBD(vv,ii,jj, 'z')
                    w_y1, w_y2 = Gradient.coordinatesFD(ww,ii,jj, 'y')
                    w_z0, w_z1 = Gradient.coordinatesBD(ww,ii,jj, 'z')

                    ux_y = Gradient.partialderivativeFD(u_y1, u_y2, y1, y2)
                    uy_y = Gradient.partialderivativeFD(v_y1, v_y2, y1, y2)
                    uz_y = Gradient.partialderivativeFD(w_y1, w_y2, y1, y2)

                    ux_z = Gradient.partialderivativeBD(u_z0, u_z1, z0, z1)
                    uy_z = Gradient.partialderivativeBD(v_z0, v_z1, z0, z1)
                    uz_z = Gradient.partialderivativeBD(w_z0, w_z1, z0, z1)

                    gradient[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])

        return gradient
