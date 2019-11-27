import numpy as np
import csv

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

        u = np.zeros([3,len(x_small)])
        u[0,:] = data['um']
        #u[1,:] = data['vm']
        #u[2,:] = data['wm']
        u_prime_x = np.zeros(len(x_small))
        for ii in range(len(x_small-1)):
            u_prime_x[ii] = (u[0,ii*len(x_small)]-u[0,ii*len(x_small)])/(x_small[2]-x_small[0])
        print(u_prime)
