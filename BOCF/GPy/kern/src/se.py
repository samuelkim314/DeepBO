# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from .kern import Kern
import scipy
from ...core.parameterization import Param
from ...core.parameterization.parameterized import Parameterized
from paramz.caching import Cache_this
from paramz.transformations import Logexp
from .rbf import RBF


class SE(Kern):
    
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='se'):
        super(SE, self).__init__(input_dim, active_dims, name)
        self.ARD = ARD
        if not ARD:
            if lengthscale is None:
                lengthscale = np.ones(1)
            else:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size in [1, input_dim], "Bad number of lengthscales"
                if lengthscale.size != input_dim:
                    lengthscale = np.ones(input_dim)*lengthscale
            else:
                lengthscale = np.ones(self.input_dim)
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size==1
        self.link_parameters(self.variance, self.lengthscale)
    
    def _to_dict(self):
        input_dict = super(SE, self)._to_dict()
        input_dict["variance"] =  self.variance.values.tolist()
        input_dict["lengthscale"] = self.lengthscale.values.tolist()
        input_dict["ARD"] = self.ARD
        return input_dict

    def K(self, X, X2=None):
        """
        Compute the kernel function.

        .. math::
            K_{ij} = k(X_i, X_j)

        :param X: the first set of inputs to the kernel
        :param X2: (optional) the second set of arguments to the kernel. If X2
                   is None, this is passed throgh to the 'part' object, which
                   handLes this as X2 == X.
        """
        if X2 is None:
            val = scipy.spatial.distance.squareform(self.variance*np.exp(-0.5*self._scaled_squared_dist(X)),checks=False)
            np.fill_diagonal(val,self.variance)
        else:
            val = self.variance*np.exp(-0.5*self._scaled_squared_dist(X, X2))
        return val

    
    def _unscaled_squared_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        if X2 is None:
            return scipy.spatial.distance.pdist(X, 'sqeuclidean')   
        else:
            return scipy.spatial.distance.cdist(X, X2, 'sqeuclidean')

    #@Cache_this(limit=3, ignore_args=())
    def _scaled_squared_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        if self.ARD:
            if X2 is not None:
                X2 = X2/self.lengthscale
            return self._unscaled_dist(X/self.lengthscale, X2)
        else:
            return self._unscaled_squared_dist(X, X2)/(self.lengthscale**2)
    
    def _scaled_squared_norm(self, D):
        """
        """
        if self.ARD:
            return np.sum(np.square(D/self.lengthscale),axis=2)      
        else:
            return np.sum(np.square(D),axis=2)/(self.lengthscale**2)
        
    def Kdiag(self, X):
        """
        The diagonal of the kernel matrix K

        .. math::
            Kdiag_{i} = k(X_i, X_i)
        """
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret
    
    def gradients_X2(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt K (dL_dK), compute the derivative wrt X
        """
        if X2 is None:
            X2 = X
        tmp =  X[:, None, :] - X2[None, :, :]
        
        if dL_dK is None:
            part1 = -self.variance*np.exp(-0.5*self._scaled_squared_norm(tmp))
            part2 = tmp/(self.lengthscale**2)
            grad = part1[:,:,None]*part2
        else:
            part1 = -(dL_dK*self.variance)*np.exp(-0.5*self._scaled_squared_norm(tmp))
            part2 = tmp/(self.lengthscale**2)
            grad = np.sum(part1[:,:,None]*part2,axis=1)
        print('test')
        print(grad)
        print(self.gradients_X2(dL_dK, X, X2))
        return grad
    
    def gradients_X(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt K (dL_dK), compute the derivative wrt X
        """
        if X2 is None: X2 = X
        aux1 =  X[:, None, :] - X2[None, :, :]
        aux2 = (-self.variance)/(self.lengthscale**2)
        if dL_dK is None:
            aux3 = np.exp((-0.5)*self._scaled_squared_norm(aux1))
            grad = (aux3[:,:,None]*aux1)*aux2
        else:
            aux3 = np.exp((-0.5)*self._scaled_squared_norm(aux1))*dL_dK
            grad = np.sum(aux3[:,:,None]*aux1,axis=1)*aux2
        return grad
    
    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)     
    
    def parameters_changed(self):
        super(SE,self).parameters_changed()    

    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Given the derivative of the objective with respect to the diagonal of
        the covariance matrix, compute the derivative wrt the parameters of
        this kernel and stor in the <parameter>.gradient field.
        See also update_gradients_full
        """
        self.variance.gradient = np.sum(dL_dKdiag)
        if not self.ARD:
            self.lengthscale.gradient = 0.
        else:
            self.lengthscale.gradient = np.zeros(self.input_dim)
        
    def update_gradients_full(self, dL_dK, X, X2=None):
        #test_kernel = RBF(input_dim = self.input_dim, variance=self.variance, lengthscale=self.lengthscale, ARD=self.ARD)
        squared_dist = self._scaled_squared_dist(X, X2)
        exp_squared_dist = np.exp(-0.5*squared_dist)
        if X2 is None:
            tmp = scipy.spatial.distance.squareform(exp_squared_dist, checks=False)
            np.fill_diagonal(tmp,1.)
            self.variance.gradient = np.sum(tmp*dL_dK)
            if self.ARD:
                self.lengthscale.gradient = (self.variance*np.sum(self.variance.gradient[:,:,None]*np.square(X[:, None, :] - X2[None, :, :]),axis=1))/(self.lengthscale**3)
            else:
                self.lengthscale.gradient = (self.variance/self.lengthscale)*np.sum(scipy.spatial.distance.squareform(exp_squared_dist*squared_dist)*dL_dK)
        else:
            self.variance.gradient = np.sum(exp_squared_dist*dL_dK)
            if self.ARD:
                self.lengthscale.gradient = (self.variance*np.sum(self.variance.gradient[:,:,None]*np.square(X[:, None, :] - X2[None, :, :]),axis=1))/(self.lengthscale**3)
            else:
                self.lengthscale.gradient = (self.variance/self.lengthscale)*np.sum((exp_squared_dist*squared_dist)*dL_dK)
        #test_kernel.update_gradients_full(dL_dK,X,X2)
        #print('test1')
        #print(test_kernel.variance.gradient)
        #print(test_kernel.lengthscale.gradient)
        #print('test2')
        #print(test_kernel.variance.gradient)
        #print(test_kernel.lengthscale.gradient)
        #print(np.abs(grad-test_grad) > 1e-8)
        