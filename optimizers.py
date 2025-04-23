import numpy as np
import scipy as sp
import geomstats
import pymanopt

class BacktrackingRGD:

    def __init__(self, manifold_class, f, df, tol = 1e-16, num_iter=100, c=0.5, g=1, lr_init=1, r=0.8):
        self.manifold_class = manifold_class
        self.m = manifold_class.m
        self.f = f
        self.df = df
        self.tol = tol
        self.num_iter = num_iter
        self.c = c
        self.g = g
        self.lr_init = lr_init 
        self.r = r
        self.alpha = lr_init
        self.res = None
        self.step_sizes = []
        self.residuals = []
        self.points = []

    def _linesearch(self, x, rgrad, f_val):
        self.alpha = self.g*self.alpha
        while self.f(self.manifold_class.exp(x,rgrad,-self.alpha)) > f_val - self.c*self.alpha*(self.res**2):
            self.alpha = self.r*self.alpha
        return self.alpha

    def train(self, x0):
        x = x0.copy()
        for i in range(self.num_iter):
            egrad = self.df(x)
            rgrad = self.manifold_class.grad(x,egrad)
            self.res = self.manifold_class.norm(x,rgrad)
            self.points.append(x)
            self.residuals.append(self.res)
            f_val = self.f(x)

            if self.res < self.tol:
                break

            self.alpha = self._linesearch(x,rgrad,f_val)

            self.step_sizes.append(self.alpha)
            x = self.manifold_class.exp(x,rgrad,-self.alpha)


class AdaptiveRGD:

    def __init__(self, manifold_class, f, df, tol = 1e-16, num_iter=100, theta = 0):
        self.manifold_class = manifold_class
        self.m = manifold_class.m
        self.f = f
        self.df = df
        self.tol = tol
        self.num_iter = num_iter
        self.theta = 0
        self.res = None
        self.step_sizes = []
        self.residuals = []
        self.points = []
        self.c = np.sqrt(2)
        self.alpha = 0.00001
        self.lda_approx = 10

    def _approximate_lipschitz(self, x, rgrad, x_new, rgrad_new,step):
        rgradnorm = self.manifold_class.norm(x,step*rgrad)
        rgraddiff = self.manifold_class.norm(x_new, rgrad_new - self.manifold_class.transport(x,x_new,rgrad))
        return rgradnorm/(self.c*rgraddiff)

    def _init_ls(self,x,rgrad):
        while self.lda_approx > 1:
            x_new = self.manifold_class.exp(x,rgrad,-self.alpha)
            rgrad_new = self.manifold_class.grad(x_new,self.df(x_new))
            self.lda_approx = self._approximate_lipschitz(x, rgrad, x_new, rgrad_new,self.alpha)
            self.alpha = 2*self.alpha

    def train(self, x0):
        x = x0.copy()
        rgrad = self.manifold_class.grad(x,self.df(x))
        self.res = self.manifold_class.norm(x,rgrad)
        self.points.append(x)
        self.residuals.append(self.res)
        self._init_ls(x,rgrad)

        for i in range(self.num_iter):
            x_old = x.copy()
            rgrad_old = rgrad.copy()
            x = self.manifold_class.exp(x,rgrad,-self.alpha)
            rgrad = self.manifold_class.grad(x,self.df(x))
            self.res = self.manifold_class.norm(x,rgrad)
            self.points.append(x)
            self.residuals.append(self.res)
            self.step_sizes.append(self.alpha)

            if self.res < self.tol:
                break

            alpha_old = self.alpha
            self.lda_approx = self._approximate_lipschitz(x_old,rgrad_old,x,rgrad,self.alpha)
            self.alpha = min(np.sqrt(1 + self.theta)*self.alpha,self.lda_approx)
            self.theta = self.alpha/alpha_old

class BWBacktrackingRGD:

    def __init__(self, manifold_class, f, df, tol = 1e-16, num_iter=100, c=0.5, g=1, lr_init=1, r=0.8):
        self.manifold_class = manifold_class
        self.m = manifold_class.m
        self.f = f
        self.df = df
        self.tol = tol
        self.num_iter = num_iter
        self.c = c
        self.g = g
        self.lr_init = lr_init 
        self.r = r
        self.alpha = lr_init
        self.res = None
        self.step_sizes = []
        self.residuals = []
        self.points = []

    def _linesearch(self, x, rgrad, f_val,product):
        self.alpha = self.g*self.alpha
        while self.f(self.manifold_class.exp(x,rgrad,-self.alpha,product)) > f_val - self.c*self.alpha*(self.res**2):
            self.alpha = self.r*self.alpha
        return self.alpha

    def train(self, x0):
        x = x0.copy()
        for i in range(self.num_iter):
            egrad = self.df(x)
            product = self.manifold_class.pointgradproduct(x,egrad)
            rgrad = self.manifold_class.grad(x,egrad)
            self.res = self.manifold_class.norm(x,rgrad)
            self.points.append(x)
            self.residuals.append(self.res)
            f_val = self.f(x)

            if self.res < self.tol:
                break

            self.alpha = self._linesearch(x,rgrad,f_val,product)

            self.step_sizes.append(self.alpha)
            x = self.manifold_class.exp(x,rgrad,-self.alpha,product)

class BWAdaptiveRGD:

    def __init__(self, manifold_class, f, df, tol = 1e-16, num_iter=100, theta = 0):
        self.manifold_class = manifold_class
        self.m = manifold_class.m
        self.f = f
        self.df = df
        self.tol = tol
        self.num_iter = num_iter
        self.theta = 0
        self.res = None
        self.step_sizes = []
        self.residuals = []
        self.points = []
        self.c = np.sqrt(2)
        self.alpha = 0.00001
        self.lda_approx = 10

    def _approximate_lipschitz(self, x, rgrad, x_new, rgrad_new,step,product):
        rgradnorm = self.manifold_class.norm(x,step*rgrad)
        rgraddiff = self.manifold_class.norm(x_new, rgrad_new - self.manifold_class.transport(x,rgrad,step,product))
        return rgradnorm/(self.c*rgraddiff)

    def _init_ls(self,x,rgrad,product):
        while self.lda_approx > 1:
            x_new = self.manifold_class.exp(x,rgrad,-self.alpha,product)
            rgrad_new = self.manifold_class.grad(x_new,self.df(x_new))
            self.lda_approx = self._approximate_lipschitz(x, rgrad, x_new, rgrad_new,self.alpha,product)
            self.alpha = 2*self.alpha

    def train(self, x0):
        x = x0.copy()
        egrad = self.df(x)
        rgrad = self.manifold_class.grad(x,egrad)
        product = self.manifold_class.pointgradproduct(x,egrad)
        self.res = self.manifold_class.norm(x,rgrad)
        self.points.append(x)
        self.residuals.append(self.res)
        self._init_ls(x,rgrad,product)

        for i in range(self.num_iter):
            x_old = x.copy()
            egrad_old = egrad.copy()
            rgrad_old = rgrad.copy()
            x = self.manifold_class.exp(x,rgrad,-self.alpha,product)
            egrad = self.df(x)
            rgrad = self.manifold_class.grad(x,egrad)
            self.res = self.manifold_class.norm(x,rgrad)
            self.points.append(x)
            self.residuals.append(self.res)
            self.step_sizes.append(self.alpha)

            if self.res < self.tol:
                break

            alpha_old = self.alpha
            self.lda_approx = self._approximate_lipschitz(x_old,rgrad_old,x,rgrad,self.alpha,product)
            self.alpha = min(np.sqrt(1 + self.theta)*self.alpha,self.lda_approx)
            self.theta = self.alpha/alpha_old
            product = self.manifold_class.pointgradproduct(x,egrad)

