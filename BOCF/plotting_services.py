# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from matplotlib import interactive
from pylab import savefig
import pylab


def plot(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:
        # X = np.arange(bounds[0][0], bounds[0][1], 0.001)
        # X = X.reshape(len(X),1)
        # acqu = acquisition_function(X)
        # acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))) # normalize acquisition
        # m, v = model.predict(X.reshape(len(X),1))
        # plt.ioff()
        # plt.figure(figsize=(10,5))
        # plt.subplot(2, 1, 1)
        # plt.plot(X, m, 'b-', label=u'Posterior mean',lw=2)
        # plt.fill(np.concatenate([X, X[::-1]]), \
        #         np.concatenate([m - 1.9600 * np.sqrt(v),
        #                     (m + 1.9600 * np.sqrt(v))[::-1]]), \
        #         alpha=.5, fc='b', ec='None', label='95% C. I.')
        # plt.plot(X, m-1.96*np.sqrt(v), 'b-', alpha = 0.5)
        # plt.plot(X, m+1.96*np.sqrt(v), 'b-', alpha=0.5)
        # plt.plot(Xdata, Ydata, 'r.', markersize=10, label=u'Observations')
        # plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        # plt.title('Model and observations')
        # plt.ylabel('Y')
        # plt.xlabel('X')
        # plt.legend(loc='upper left')
        # plt.xlim(*bounds)
        # grid(True)
        # plt.subplot(2, 1, 2)
        # plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        # plt.plot(X,acqu_normalized, 'r-',lw=2)
        # plt.xlabel('X')
        # plt.ylabel('Acquisition value')
        # plt.title('Acquisition function')
        # grid(True)
        # plt.xlim(*bounds)

        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.01)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        m, v = model.predict(x_grid)


        model.plot_density(bounds[0], alpha=.5)

        plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
        plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
        plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

        plt.plot(Xdata, Ydata, 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))

        plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition (arbitrary units)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.legend(loc='upper left')


        if filename!=None:
            savefig(filename)
        else:
            plt.show()

    if input_dim ==2:
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(X)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, m.reshape(200,200),100)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Posterior sd.')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'k.', markersize=10)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        if filename!=None:
            savefig(filename)
        else:
            plt.show()
            
def integrated_plot(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample, attribute=0, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:
        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.01)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        mean, var = model.predict_noiseless(x_grid)
        
        #m = mean[attribute,:]
        #v = v[attribute,:]

        output_dim = len(Ydata)
        
        #figures = [None]*output_dim
        #interactive(True)
        for j in range(1):
            m = mean[j,:]
            v = var[j,:]
            plt.figure()
            plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6, label ='posterior mean')
            plt.plot(x_grid, m-1.96*np.sqrt(v), 'g--', alpha = 0.2, label ='confidence interval')
            plt.plot(x_grid, m+1.96*np.sqrt(v), 'g--', alpha=0.2)
    
            plt.plot(Xdata, Ydata[j], 'r.', markersize=10)
            plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
            factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))
    
            plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='acquisition (arbitrary units)')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.25*factor)
            plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
            plt.legend(loc='upper left')

    elif input_dim == 2:
        n = 20
        X1 = np.linspace(bounds[0][0], bounds[0][1], n)
        X2 = np.linspace(bounds[1][0], bounds[1][1], n)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(n*n,1),x2.reshape(n*n,1)))
        acqu = acquisition_function(X)
        #suggested_sample = np.atleast_2d(X[np.argmax(acqu.flatten()),:])
        print(acqu)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        print(acqu_normalized)
        acqu_normalized = acqu_normalized.reshape((n,n))
        mean, var = model.predict_noiseless(X)
        m = mean[0, :]
        v = var[0, :]
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, m.reshape(n,n),100)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label='Observations')
        plt.colorbar()
        plt.xlabel('d')
        plt.ylabel('theta')
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.legend(loc='upper left')
        ##
        plt.subplot(1, 3, 2)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label='Observations')
        plt.contourf(X1, X2, v.reshape(n,n),100)
        plt.colorbar()
        plt.xlabel('d')
        plt.ylabel('theta')
        plt.title('Posterior variance')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.legend(loc='upper left')
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'r*', markersize=10, label='Suggested point to evaluate')
        plt.xlabel('d')
        plt.ylabel('theta')
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.legend(loc='upper left')
    
    if filename!=None:
        savefig(filename)
    else:
        plt.show()
        

def plot_acquisition(bounds, input_dim, acquisition_function, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:

        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        plt.plot(x_grid, acqu_normalized, 'r-',lw=2,label ='Acquisition function')
        plt.xlabel('x')
        plt.ylabel('a(x)')
        #plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
    elif input_dim == 2:
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        plt.plot()
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        #plt.plot(suggested_sample[:,0],suggested_sample[:,1],'r*', markersize=10)
        plt.xlabel('d')
        plt.ylabel('theta')
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.legend(loc='upper left')
    if filename!=None:
        savefig(filename)
    else:
        plt.show()


def plot_convergence(historic_optimal_values, var_at_historical_optima=None, confidence_interval=False, filename=None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
    n = len(historic_optimal_values)

    # Estimated m(x) at the proposed sampling points
    #plt.subplot(1, 2, 2)
    plt.plot(list(range(n)), historic_optimal_values,'-o')
    if confidence_interval:
        plt.plot(list(range(n)), historic_optimal_values-1.96*np.sqrt(var_at_historical_optima), 'k-', alpha = 0.2)
        plt.plot(list(range(n)), historic_optimal_values+1.96*np.sqrt(var_at_historical_optima), 'k-', alpha = 0.2)
        factor = max(historic_optimal_values+1.96*np.sqrt(var_at_historical_optima))-min(historic_optimal_values-1.96*np.sqrt(var_at_historical_optima))
        plt.ylim(min(historic_optimal_values-1.96*np.sqrt(var_at_historical_optima))-0.25*factor,  max(historic_optimal_values+1.96*np.sqrt(var_at_historical_optima))+0.05*factor)
    plt.title('Expected value of historical optimal points given the true attributes')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    grid(True)

    if filename!=None:
        savefig(filename)
    else:
        plt.show()


def plot_pareto_front_comparison(estimated_pareto_front, true_pareto_front=None, approximately=True):
    plt.figure()
    if true_pareto_front is not None:
        if approximately:
            label = 'True Pareto front (approximately)'
        else:
            label = 'True Pareto front'
        plt.plot(true_pareto_front[0, :], true_pareto_front[1, :], 'ko', label=label)
    plt.plot(estimated_pareto_front[0, :], estimated_pareto_front[1, :], 'ro', label='Estimated Pareto front')
    plt.xlabel('f_1')
    plt.ylabel('f_2')
    plt.title('ParEGO; noiseless observations')
    plt.legend(loc='lower left')
    plt.show()
