import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m
from collections import defaultdict

class Read:

    object_counter = 0

    def __init__(self):
        self.object_id = Read.object_counter
        Read.object_counter += 1

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

    def importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_component):
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

    def importStressTensor(usecase, DIM_Y, DIM_Z):
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

        for ii in range(DIM_Y):
#            for jj in range(DIM_Z):
            tensor[ii,:,0,0] = data['uu'][ii]
            tensor[ii,:,1,0] = data['uv'][ii]
            tensor[ii,:,2,0] = data['uw'][ii]

            tensor[ii,:,0,1] = data['uv'][ii]
            tensor[ii,:,1,1] = data['vv'][ii]
            tensor[ii,:,2,1] = data['vw'][ii]

            tensor[ii,:,0,2] = data['uw'][ii]
            tensor[ii,:,1,2] = data['vw'][ii]
            tensor[ii,:,2,2] = data['ww'][ii]

        return tensor
