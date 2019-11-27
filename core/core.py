import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl


class Core:

    #counter = 0

    def __init__(self):
        self.object_id = Core.counter
        #Core.counter += 1

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
        print(i)
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


    def calc_gradient(self):
        data = self.data
        x = data['Y']
        y = data['Z']


        x_small = np.zeros(len(x))
        y_small = np.zeros(len(y))
        counter=0
        for ii in range(len(x)-1):
            if (x[ii] != x[ii+1]):
                x_small[counter] = x[ii]
                counter+=1

        counter=0
        for ii in range(len(y)-1):
            if (y[ii] != y[ii+1]):
                y_small[counter] = y[ii]
                counter+=1
            
        x_small = x_small[0:counter+1]
        
        yy = np.reshape(data['Y'], [129,129])
        zz = np.reshape(data['Z'], [129,129])
        uu = np.reshape(data['um'],[129,129])
        uv = np.reshape(data['vm'],[129,129])
        uw = np.reshape(data['wm'],[129,129])
    
        
        u = np.array([uu,uv,uw])
    
        dx = 0.1
        dy = 0.1
        dz = 0.1
        print(np.shape(u))
        nabla_u = np.gradient(u,dx)
        print(np.shape(nabla_u[1]))
        
        gradient = np.zeros([129,129,3,3])
        for ii in range(0):
            for jj in range(0):
                for nn in range(3):
                    for mm in range(3):
                        gradient[ii,jj,nn,mm] = nabla_u[nn][mm,ii,jj]
                
        print(np.shape(gradient))
        fig = plt.figure()
        ax = fig.gca()
        ax.quiver(u[0,:], u[1,:], dx, dy, color='r',
                  angles='xy', scale_units='xy')
        plt.show()
    
