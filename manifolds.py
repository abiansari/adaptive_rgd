import numpy as np
from pymanopt.manifolds import stiefel, sphere, grassmann
from geomstats.geometry import spd_matrices

class Stiefel:

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.m = stiefel.Stiefel(self.n,self.p)

    def grad(self,x,e_grad):
        return self.m.egrad2rgrad(x,e_grad)
    
    def norm(self,x,v):
        return self.m.norm(x,v)

    def transport(self,x_old,x,v):
        return self.m.transp(x_old,x,v)
    
    def exp(self,x,v,alpha):
        return self.m.exp(x,alpha*v)

class Sphere:

    def __init__(self, n):
        self.n = n
        self.m = sphere.Sphere(self.n)

    def grad(self,x,e_grad):
        return self.m.egrad2rgrad(x,e_grad)
    
    def norm(self,x,v):
        return self.m.norm(x,v)

    def transport(self,x_old,x,v):
        return self.m.transp(x_old,x,v)
    
    def exp(self,x,v,alpha):
        return self.m.exp(x,alpha*v)

class Grassmannian:

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.m = grassmann.Grassmann(self.n,self.p)

    def grad(self,x,e_grad):
        return self.m.egrad2rgrad(x,e_grad)
    
    def norm(self,x,v):
        return self.m.norm(x,v)

    def transport(self,x_old,x,v):
        return self.m.transp(x_old,x,v)
    
    def exp(self,x,v,alpha):
        return self.m.exp(x,alpha*v)

class BuresWassersteinSPD:

    def __init__(self, n):
        self.n = n
        self.m = spd_matrices.SPDMetricBuresWasserstein(self.n)

    def grad(self,x,e_grad):
        return x@e_grad + e_grad@x
    
    def pointgradproduct(self,x,egrad):
        return egrad@x@egrad
    
    def norm(self,x,v):
        return self.m.norm(v,x)
    
    def transport(self,x,v,alpha,product):
        return v + 2*alpha*product
    
    def exp(self,x,v,alpha,product):
        return x + alpha*v + (alpha**2)*product
