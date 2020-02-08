import numpy as np
import matplotlib.pyplot as plt
import math as m
import Read

class Plot:

    def plotMeanVelocityComponent(RA, Retau, velocity_component):

        fig = plt.figure(1,figsize=(10,30))
        gs = fig.add_gridspec(6, 14)
        fig.suptitle('Mean velocity flow for $R_{\\tau}$ and different RA values for velocity compoment '+ velocity_component)
        fig.subplots_adjust(hspace = 1)
        for ii in range(len(RA)):
            ra = RA[ii]
            ncols = RA[ii]-1

            ### Collect coordinates and mean velocity data
            usecase = str(ra)+'_'+str(Retau)

            zcoord, DIM_Z = Read.importCoordinates('z', usecase)
            ycoord, DIM_Y = Read.importCoordinates('y', usecase)

            U = Read.importMeanVelocity(DIM_Y, DIM_Z, usecase, velocity_component)

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

    def plotMeanVelocityField(RA, Retau, data, ycoord, zcoord):
        x, y, z = np.meshgrid(ycoord[1:-1:10], zcoord[1:-1:10], np.zeros(1))

        fig_field = plt.figure(2)
        ax = fig_field.gca(projection='3d')
        ax.quiver(x, y, z, data[1:-1:10,1:-1:10,0], data[1:-1:10,1:-1:10,1], data[1:-1:10,1:-1:10,2], length=0.05, normalize=True)
        ax.set_title('Mean Velocity field for $R_{\\tau}$ = '+str(Retau) +' and RA = '+str(RA))

    def tensorplot(tensor, DIM_Y, DIM_Z, title=None):
        tensor = np.reshape(tensor, (DIM_Y, DIM_Z, 9))
        
        ## Rescale each vector element t0 [0,1]
        for i in range(0,9):
            tensor[:,:,i] -= tensor[:,:,i].min()
            tensor[:,:,i] /= tensor[:,:,i].max()/0.006
        
        fig, axes = plt.subplots(nrows=3, ncols=3)
        fig.suptitle(title)
        ii = 0
        for ax in axes.flat:
            im = ax.imshow(tensor[:,:,ii])
            ax.plot([int(DIM_Y/2),int(DIM_Y/2)],[0,DIM_Z], 'w-')
            ax.set_xlim([0, DIM_Y])
            ax.set_ylim([0, DIM_Z])
            
            if ii not in [0,3,6]:
                ax.get_yaxis().set_visible(False)
            
            if ii not in [6,7,8]:
                ax.get_xaxis().set_visible(False)
            
            ii += 1
            
        plt.subplots_adjust(wspace=0.125, hspace=0.125,top=0.92, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        fig2, axes2 = plt.subplots(nrows=3, ncols=3)
        fig2.suptitle(title)
        ii = 0
        for ax in axes2.flat:
            im = ax.imshow(tensor[:,:,ii])
            ax.plot([0,DIM_Y], [int(DIM_Z/2),int(DIM_Z/2)], 'w-')
            ax.set_xlim([0, DIM_Y])
            ax.set_ylim([0, DIM_Z])
            
            if ii not in [0,3,6]:
                ax.get_yaxis().set_visible(False)
            
            if ii not in [6,7,8]:
                ax.get_xaxis().set_visible(False)
                
            ii += 1
            
        plt.subplots_adjust(wspace=0.125, hspace=0.125,top=0.92, right=0.8)
        cbar_ax1 = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
        fig2.colorbar(im, cax=cbar_ax1)
        
        fig3, axes3 = plt.subplots(nrows=3, ncols=3)
        fig3.suptitle(title)
        ii=0
        for ax in axes3.flat:
            ax.plot(tensor[:, int(DIM_Z/2), ii], np.arange(DIM_Y))
            ax.set_ylim([0, DIM_Z])
            ax.set_xlim([0, 0.006])
            if ii not in [0,3,6]:
                ax.get_yaxis().set_visible(False)
            
            if ii not in [6,7,8]:
                ax.get_xaxis().set_visible(False)
                
            ii += 1
            
        plt.subplots_adjust(wspace=0.125, hspace=0.125,top=0.92)
            
        fig4, axes4 = plt.subplots(nrows=3, ncols=3)
        fig4.suptitle(title)
        ii=0
        for ax in axes4.flat:
            ax.plot(np.arange(DIM_Y), tensor[:, int(DIM_Z/2), ii])
            ax.set_xlim([0, DIM_Y])
            ax.set_ylim([0, 0.006])
            
            if ii not in [0,3,6]:
                ax.get_yaxis().set_visible(False)
            
            if ii not in [6,7,8]:
                ax.get_xaxis().set_visible(False)
                
            ii += 1
        
        plt.subplots_adjust(wspace=0.125, hspace=0.125,top=0.92)