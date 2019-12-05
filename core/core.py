import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as m

class Core:
    
    object_counter = 0
        
    def __init__(self):
        self.object_id = Core.object_counter
        Core.object_counter += 1
    
    def importCoordinates(coordinate, usecase):
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

    def importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_component):
        filepath = '../inversion/DATA/RECTANGULARDUCT/DATA/'+velocity_component+'_' + usecase + '.prof.txt'    
        U = np.zeros([DIM_Y, DIM_Z])
    
        counter = 0
        with open(filepath, encoding='latin-1') as fd:
            for i in range(24):
                next(fd)
            for line in fd:
                dataline= line.split()
                if len(dataline) == DIM_Z:
                    U[counter,:] = dataline
                    counter += 1
                
        fd.close()
        
        return U
    
    def plotMeanVelocity(self, RA, Retau, velocity_component):
        
        fig = plt.figure(figsize=(10,30))
        gs = fig.add_gridspec(6, 14)
        fig.suptitle('Mean velocity flow for $R_{\\tau}$ and different RA values for velocity compoment '+ velocity_component)
        fig.subplots_adjust(hspace = 1)
        for ii in range(len(RA)):
            ra = RA[ii]
            ncols = RA[ii]-1
            
            ### Collect coordinates and mean velocity data
            usecase = str(ra)+'_'+str(Retau)
            
            zcoord, DIM_Z = Core.importCoordinates('z', usecase)
            
            if ra == 1: ## In case of a square grid, y and z are identical
                ycoord = zcoord
                DIM_Y = DIM_Z
            else:
                ycoord, DIM_Y = Core.importCoordinates('y', usecase)
            
            U = Core.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_component) 
            
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
                
                
            
        
        plt.show()
        
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

    def calc_S_R(grad_u, k, eps, n):

        S = np.zeros((n, 3, 3))
        R = np.zeros((n, 3, 3))
        for i in range(n):
            S[i, :, :] = k[i]/eps[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            R[i, :, :] = k[i]/eps[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        return S,R


###################### Gradient ##################################
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
        
        ## As the flow is symmetric in the x-direction, the gradient in x-direction is always zero.
        ux_x = 0
        uy_x = 0
        uz_x = 0
        
        for ii in range(grid_size):
            for jj in range(grid_size):
                
                y1 = yy[ii,jj]
                z1 = zz[ii,jj]
                
                ## Inside grid points
                if ((ii > 0 and ii < grid_size-2) and (jj > 0 and jj < grid_size-2)):
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
                
                ## Upper boundary --> z = 0
                if ((ii > 0 and ii < grid_size-2) and (jj == 0)):
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
                
                ## Lower boundary --> z = grid_size-1
                if ((ii > 0 and ii < grid_size-2) and (jj == grid_size-1)):
                    
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
                    
                ## Left boundary --> y = 0
                if ((ii == 0) and (jj > 0 and jj < grid_size-2)):
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
                    
                ## Right boundary --> y = grid_size - 1
                if ((ii == grid_size-1) and (jj > 0 and jj < grid_size-2)):
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])
                    
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])    
                    
                ## Right-top corner --> y = grid_size-1, z = 0
                if ((ii == grid_size-1) and (jj == 0)):
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])    
                    
                ## Right-bottom corner --> y = grid_size-1, z = grid_size-1
                if ((ii == grid_size-1) and (jj == grid_size-1)):
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]])    
                    
                ## Left-bottom corner --> y = 0, z = grid_size-1
                if ((ii == 0) and (jj == grid_size-1)):
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
                    
                    gradient_manual[ii,jj,:,:] = np.array([[ux_x, ux_y, ux_z], [uy_x, uy_y, uy_z], [uz_x, uz_y, uz_z]]) 
        
##########   CODE FOR TESTING: PRINTS TENSOR FOR EACH DOMAIN   ##########
#        print('Tensor at a normal grid point:')
#        print(gradient_manual[1,1,:,:])
#        print('')
#        
#        print('Tensor at a upper boundary grid point:')
#        print(gradient_manual[1,0,:,:])            
#        print('')
#        
#        print('Tensor at a lower boundary grid point:')            
#        print(gradient_manual[1,grid_size-1,:,:])
#        print('')
#        
#        print('Tensor at a left boundary grid point:')            
#        print(gradient_manual[0,1,:,:])
#        print('')
#        
#        print('Tensor at a right boundary grid point:')            
#        print(gradient_manual[grid_size-1,1,:,:])
#        print('')
#        
#        print('Tensor at top-left corner grid point:')            
#        print(gradient_manual[0,0])
#        print('')
#        
#        print('Tensor at top-right corner grid point:')            
#        print(gradient_manual[grid_size-1,0])
#        print('')
#        
#        print('Tensor at bottom-right corner grid point:')            
#        print(gradient_manual[grid_size-1,grid_size-1])
#        print('')
#        
#        print('Tensor at bottom-left corner grid point:')            
#        print(gradient_manual[0,grid_size-1])
#        print('')
#        
                    
                    