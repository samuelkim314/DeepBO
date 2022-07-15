# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import cma


class Optimizer(object):
    """
    Class for a general acquisition optimizer.

    :param bounds: list of tuple with bounds of the optimizer
    """

    def __init__(self, bounds):
        self.bounds = bounds

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        raise NotImplementedError("The optimize method is not implemented in the parent class.")


class OptSGD(Optimizer):
    '''
    (Stochastic) gradient descent algorithm.
    '''
    def __init__(self, bounds, maxiter=160):
        super(OptSGD, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        x = np.copy(x0)
        x_opt = np.copy(x0)
        #print('initial point')
        #print(x0)
        #print('initial value')
        f_opt = f(x_opt)
        #print(f_opt)
        for t in range(1, self.maxiter + 1):
            if t % 30 == 0:
                val = f(x)
                if val < f_opt:
                    x_opt = np.copy(x)
                    f_opt = np.atleast_2d(np.copy(val))
                    #print('better value was found at iteration {}'.format(t))
                    #print('point')
                    #print(x_opt)
                    #print('value')
                    #print(f_opt)
                       
            grad = f_df(x)[1]
            if not np.isnan(grad).any():
                if t < self.maxiter - 49:
                    x = x - 0.05 * grad
                else:
                    x = x - 0.05 * np.power(t + 50 -  self.maxiter, -0.7) * grad
            else:
                print('nan found')
            for k in range(x.shape[1]):
                if x[0,k] < self.bounds[k][0]:
                    x[0,k] = self.bounds[k][0]
                elif x[0,k] > self.bounds[k][1]:
                    x[0,k] = self.bounds[k][1]
            if True:
                print('test begin')
                print(f_df(x)[1])
                fx = f(x)
                h = 1e-6
                x[0,0] +=h
                f_aux = f(x)
                print((f_aux-fx)/h)
                x[0,0] -=h
                x[0,2] +=h
                f_aux = f(x)
                print((f_aux-fx)/h)
                x[0,2] -=h
                print('test end')
            #if np.absolute(fx - f_previous) < 1e-5:
                #break       
        
        x = np.atleast_2d(x)
        fx = np.atleast_2d(f(x))
        if fx < f_opt:
            x_opt = x
            f_opt = fx
        #print('final point')
        #print(x_opt)
        #print('final value')
        #print(f_opt)
        #print('initial point again')
        #print(x0)
        #print('initial value again')
        #print(f(x0))
        return x_opt, f_opt
    
    
class OptADAM(Optimizer):
    '''
    ADAM algorithm.
    '''
    def __init__(self, bounds, maxiter=100):
        super(OptADAM, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        x = x0
        alpha = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = 0*x0
        v = 0*x0
        beta1_power = 1.
        beta2_power = 1.

        for t  in range(1, self.maxiter + 1):
            #print(t)
            f_x, g_x = f_df(x)
            m = beta1*m +(1-beta1)*g_x
            v = beta2*v +(1-beta2)*np.square(g_x)
            beta1_power = beta1_power*beta1
            m_hat = m/(1-beta1_power)
            beta2_power = beta2_power*beta2
            v_hat = v/(1-beta2_power)
            tmp = alpha*np.divide(m_hat,np.sqrt(v_hat)+eps)
            if np.any(np.isnan(tmp)):
                print('nan found')
                x = np.atleast_2d(x)
                f_x = f(x)
                return x, f_x
            
            x = x - tmp
            for k in range(x.shape[1]):
                if x[0,k] < self.bounds[k][0]:
                    x[0,k] = self.bounds[k][0]
                elif x[0,k] > self.bounds[k][1]:
                    x[0,k] = self.bounds[k][1]

        x = np.atleast_2d(x)      
        f_x = f(x)
        return x, f_x


class OptAMSGrad(Optimizer):
    '''
    AMSGrad algorithm.
    '''

    def __init__(self, bounds, maxiter=120):
        super(OptAMSGrad, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """

        x_opt = np.copy(x0)
        x_ref = np.copy(x0)
        #print('initial point')
        #print(x_opt)
        #print('initial value')
        f_opt = f(x_opt)
        #print(f_opt)
        x = np.copy(x0)
        alpha = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = 0 * x0
        v = 0 * x0
        v_hat = 0 * x0

        for t in range(1, self.maxiter + 1):
            if t % 30 == 0 or np.max(x-x_ref) > 0.1:
                x_ref = np.copy(x)
                val = f(x)
                if val < f_opt:
                    x_opt = np.copy(x)
                    f_opt = np.atleast_2d(np.copy(val))
                    #print('better value was found at iteration {}'.format(t))
                    #print('point')
                    #print(x_opt)
                    #print('value')
                    #print(f_opt)
            g_x = f_df(x)[1]

            if not np.isnan(g_x).any():
                m = beta1 * m + (1 - beta1) * g_x
                v = beta2 * v + (1 - beta2) * np.square(g_x)
                v_hat = np.maximum(v_hat, v)
                tmp = alpha * np.divide(m, np.sqrt(v_hat) + eps)
                if t > self.maxiter - 50:
                    tmp /= np.sqrt(t -self.maxiter + 50)
                if not np.isnan(tmp).any():
                    x = x - tmp
                else:
                    return x_opt, f_opt

            for k in range(x.shape[1]):
                if x[0, k] < self.bounds[k][0]:
                    x[0, k] = self.bounds[k][0]
                elif x[0, k] > self.bounds[k][1]:
                    x[0, k] = self.bounds[k][1]

        x = np.atleast_2d(x)
        fx = np.atleast_2d(f(x))
        if fx < f_opt:
            x_opt = x
            f_opt = fx
        #print('final point')
        #print(x_opt)
        #print('final value')
        #print(f_opt)
        #print('initial point again')
        #print(x0)
        #print('initial value again')
        #print(f(x0))
        return x_opt, f_opt


class OptAGD(Optimizer):
    '''
    ADAM algorithm.
    '''
    def __init__(self, bounds, maxiter=250):
        super(OptAGD, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        x = x0
        y = np.copy(x0)
        lambd = 0
        gamma = 1
        beta_inverse = 100

        for t  in range(1, self.maxiter + 1):
            f_x, g_x = f_df(x)
            tmp_y = x - beta_inverse*g_x
            x = (1 - gamma)*tmp_y + gamma*y
            y =  np.copy(tmp_y)
            tmp_lamb  = (1 + np.sqrt(1 + 4*lambd))/2
            gamma = (1 - lambd)/tmp_lamb
            lamb = np.copy(tmp_lamb)

            for k in range(x.shape[1]):
                if x[0,k] < self.bounds[k][0]:
                    x[0,k] = self.bounds[k][0]
                elif x[0,k] > self.bounds[k][1]:
                    x[0,k] = self.bounds[k][1]
                    
        x = np.atleast_2d(x)      
        f_x = f_df(x)[0]
        return x, f_x
    

class OptLbfgs(Optimizer):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients.
    '''
    def __init__(self, bounds, maxiter=500):
        super(OptLbfgs, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        import scipy.optimize
        if f_df is None and df is not None:
            f_df = lambda x: float(f(x)), df(x)
            
        if f_df is None and df is None:
            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.bounds, approx_grad=True, maxiter=self.maxiter, factr=1e3, pgtol=1e-20)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(f_df, x0=x0, bounds=self.bounds, maxiter=self.maxiter, factr=1e6)

        ### --- We check here if the the optimizer moved. It it didn't we report x0 and f(x0) as scipy can return NaNs
        if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            result_x  = np.atleast_2d(x0)
            result_fx =  np.atleast_2d(f(x0))
        else:
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])
            
        #print(res)
        return result_x, result_fx


class OptLbfgs2(Optimizer):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients.
    '''

    def __init__(self, bounds, maxiter=50):
        super(OptLbfgs2, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        import scipy.optimize
        if f_df is None and df is not None:
            f_df = lambda x: float(f(x)), df(x)

        if f_df is None and df is None:
            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.bounds, approx_grad=True, maxiter=self.maxiter,
                                               factr=1e6)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(f_df, x0=x0, bounds=self.bounds, maxiter=self.maxiter, factr=1e5, pgtol=1e-15)

        ### --- We check here if the the optimizer moved. It it didn't we report x0 and f(x0) as scipy can return NaNs
        if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            result_x = np.atleast_2d(x0)
            result_fx = np.atleast_2d(f(x0))
        else:
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])

        #print(res)
        return result_x, result_fx


class OptDirect(Optimizer):
    '''
    Wrapper for DIRECT optimization method. It works partitioning iteratively the domain
    of the function. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=50):
        super(OptDirect, self).__init__(bounds)
        self.maxiter = maxiter
        #assert self.space.has_types['continuous']

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        # Based on the documentation of DIRECT, it does not seem we can pass through an initial point x0
        try:
            from DIRECT import solve
            def DIRECT_f_wrapper(f):
                def g(x, user_data):
                    return f(np.array([x])), 0
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=self.maxiter)
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find DIRECT library, please install it to use this option.")


class OptCma(Optimizer):
    '''
    Wrapper the Covariance Matrix Adaptation Evolutionary strategy (CMA-ES) optimization method. It works generating
    an stochastic search based on multivariate Gaussian samples. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=50):
        super(OptCma, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        try:
            import cma
            def CMA_f_wrapper(f):
                def g(x):
                    return f(np.array([x]))[0][0]
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x = cma.fmin(CMA_f_wrapper(f), x0, 0.6, options={"bounds":[lB, uB], "verbose":-1, "maxfevals":100})[0]
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find cma library, please install it to use this option.")
        except:
            print("CMA does not work in problems of dimension 1.")


def apply_optimizer(optimizer, x0, f=None, df=None, f_df=None, duplicate_manager=None, context_manager=None, space=None):
    """
    :param x0: initial point for a local optimizer (x0 can be defined with or without the context included).
    :param f: function to optimize.
    :param df: gradient of the function to optimize.
    :param f_df: returns both the function to optimize and its gradient.
    :param duplicate_manager: logic to check for duplicate (always operates in the full space, context included)
    :param context_manager: If provided, x0 (and the optimizer) operates in the space without the context
    :param space: GPyOpt class design space.
    """

    x0 = np.atleast_2d(x0)

    ## --- Compute a new objective that inputs non context variables but that takes into account the values of the context ones.
    ## --- It does nothing if no context is passed
    #problem = OptimizationWithContext(x0=x0, f=f, df=df, f_df=f_df, context_manager=context_manager)

    #if context_manager:
        #print('context manager')
        #add_context = lambda x : context_manager._expand_vector(x)
    #else:
        #add_context = lambda x : x

    #if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(x0):
        #raise ValueError("The starting point of the optimizer cannot be a duplicate.")

    ## --- Optimize point
    #optimized_x, suggested_fx = optimizer.optimize(problem.x0_nocontext, problem.f_nocontext, problem.df_nocontext, problem.f_df_nocontext)
        
    ## --- Add context and round according to the type of variables of the design space
    #suggested_x_with_context = add_context(optimized_x)
    #suggested_x_with_context_rounded = space.round_optimum(suggested_x_with_context)

    ## --- Run duplicate_manager
    #if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(suggested_x_with_context_rounded):
        #suggested_x, suggested_fx = x0, np.atleast_2d(f(x0))
    #else:
        #suggested_x, suggested_fx = suggested_x_with_context_rounded, f(suggested_x_with_context_rounded)
    suggested_x, suggested_fx = optimizer.optimize(x0, f, df, f_df)
    #suggested_fx = f(suggested_x)

    return suggested_x, suggested_fx


def apply_optimizer_inner(optimizer, x0, f=None, df=None, f_df=None, duplicate_manager=None, context_manager=None, space=None):
    """
    :param x0: initial point for a local optimizer (x0 can be defined with or without the context included).
    :param f: function to optimize.
    :param df: gradient of the function to optimize.
    :param f_df: returns both the function to optimize and its gradient.
    :param duplicate_manager: logic to check for duplicate (always operates in the full space, context included)
    :param context_manager: If provided, x0 (and the optimizer) operates in the space without the context
    :param space: GPyOpt class design space.
    """

    x0 = np.atleast_2d(x0)
    #print('apply inner opt')

    ## --- Compute a new objective that inputs non context variables but that takes into account the values of the context ones.
    ## --- It does nothing if no context is passed
    #problem = OptimizationWithContext(x0=x0, f=f, df=df, f_df=f_df, context_manager=context_manager)

    #if context_manager:
        #print('context manager')
        #add_context = lambda x : context_manager._expand_vector(x)
    #else:
        #add_context = lambda x : x

    #if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(x0):
        #raise ValueError("The starting point of the optimizer cannot be a duplicate.")

    ## --- Optimize point
    #optimized_x, suggested_fx = optimizer.optimize(problem.x0_nocontext, problem.f_nocontext, problem.df_nocontext, problem.f_df_nocontext)
        
    ## --- Add context and round according to the type of variables of the design space
    #suggested_x_with_context = add_context(optimized_x)
    #suggested_x_with_context_rounded = space.round_optimum(suggested_x_with_context)

    ## --- Run duplicate_manager
    #if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(suggested_x_with_context_rounded):
        #suggested_x, suggested_fx = x0, np.atleast_2d(f(x0))
    #else:
        #suggested_x, suggested_fx = suggested_x_with_context_rounded, f(suggested_x_with_context_rounded)
    suggested_x, suggested_fx = optimizer.optimize(x0, f, df, f_df)
    #suggested_fx = f(suggested_x)

    return suggested_x, suggested_fx


def optimize_anchor_points(id, optimizer, anchor_points, f=None, df=None, f_df=None, duplicate_manager=None, context_manager=None, space=None):
    return [apply_optimizer(optimizer, a, f, df, f_df, duplicate_manager, context_manager, space) for a in anchor_points]


class OptimizationWithContext(object):

    def __init__(self, x0, f, df=None, f_df=None, context_manager=None):
        '''
        Constructor of an objective function that takes as input a vector x of the non context variables
        and retunrs a value in which the context variables have been fixed.
        '''
        self.x0 = np.atleast_2d(x0)
        self.f = f
        self.df = df
        self.f_df = f_df
        self.context_manager = context_manager

        if not context_manager:
            self.x0_nocontext = x0
            self.f_nocontext  =  self.f
            self.df_nocontext  =  self.df
            self.f_df_nocontext = self.f_df

        else:
            #print('context')
            self.x0_nocontext = self.x0[:,self.context_manager.noncontext_index]
            self.f_nocontext  = self.f_nc
            if self.f_df is None:
                self.df_nocontext = None
                self.f_df_nocontext = None
            else:
                self.df_nocontext = self.df
                self.f_df_nocontext  = self.f_df#self.f_df_nc

    def f_nc(self,x):
        '''
        Wrapper of *f*: takes an input x with size of the noncontext dimensions
        expands it and evaluates the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        if x.shape[0] == 1:
            return self.f(xx)[0]
        else:
            return self.f(xx)

    def df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        _, df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return df_nocontext_xx

    def f_df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        f_nocontext_xx , df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return f_nocontext_xx, df_nocontext_xx


def choose_optimizer(optimizer_name, bounds):
        """
        Selects the type of local optimizer
        """          
        if optimizer_name == 'lbfgs':
            optimizer = OptLbfgs(bounds)

        elif optimizer_name == 'lbfgs2':
            optimizer = OptLbfgs2(bounds)

        elif optimizer_name == 'sgd':
            optimizer = OptSGD(bounds)
            
        elif optimizer_name == 'adam':
            optimizer = OptADAM(bounds)

        elif optimizer_name == 'amsgrad':
            optimizer = OptAMSGrad(bounds)

        elif optimizer_name == 'DIRECT':
            optimizer = OptDirect(bounds)

        elif optimizer_name == 'CMA':
            optimizer = OptCma(bounds)

        elif optimizer_name == 'agd':
            optimizer = OptAGD(bounds)
        else:
            raise InvalidVariableNameError('Invalid optimizer selected.')

        return optimizer
