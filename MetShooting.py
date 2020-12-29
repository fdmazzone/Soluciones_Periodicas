#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:40:32 2020

@author: fernando
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint , simps
import scipy.optimize
from matplotlib import cm
import numpy as np
from numpy import linalg as LA

###############################################################################
###############  Blobal Variables   ###########################################
###############################################################################
psi=lambda p:.5*(-1+np.sqrt(1+4*np.abs(p)))*np.sign(p)
alpha=1/2.0
gamma=1/3.5
T=2*np.pi
#####################  Potential Gradient #####################################
def nabla_F(t,x): 
    R=x**2 + 1
    return -2*x*R**(alpha - 1)*\
        (alpha*np.sin(R**gamma) + gamma*R**gamma*np.cos(R**gamma))+np.cos(t)

################ Hamiltonian System ###########################################
def ecuacion_hamilton(y,t):
    x,p=y
    dxdt=psi(p)
    dpdt= nabla_F(t,x)
    return dxdt,dpdt

#####################  Mapeo de Poincare  #####################################
def PoincareMap(x):
        t = np.array([0,T])
        sol = odeint(ecuacion_hamilton,x ,t)
        return sol[-1,:]

Error_PM = lambda x: (PoincareMap(x)-x).dot(PoincareMap(x)-x)

###############################################################################
############ Color Map Poincare map ###########################################
###############################################################################


def GrafPoincareError(rangos):
    x0,v0=rangos
    t = np.array([0,T])
    X0,V0=np.meshgrid(x0,v0)
    n,k=np.shape(X0)
    error=np.zeros([n,k])
    
    for i in range(n):
        for j in range(k):
            error[i,j]=Error_PM([X0[i,j],V0[i,j]])
    return X0,V0, error
    
x0=np.arange(-50,50,.1)
v0=np.arange(-5,5,.1)
X0,V0,error=GrafPoincareError([x0,v0])

fig = plt.figure()
ax=fig.add_axes([0.1,.1,.8,.8])
niveles=np.arange(0,1,.2)
#s=ax.contourf(X0, V0, error, niveles, cmap='RdGy')
ax.contour(X0, V0, error, niveles)
s=ax.imshow(error, extent=[-50, 50, -5, 5], origin='lower',
           cmap='RdGy')

ax.set_aspect(aspect=3)


fig.colorbar(s, ax=ax)
ax.set_xlabel(r"$x(0)$",fontsize=16)
ax.set_ylabel(r"$x'(0)$",fontsize=16) 


###############################################################################
############ Finding fixed point  #############################################
###############################################################################

rangos=((15.5,16.5), (-1,1))
opt=scipy.optimize.dual_annealing(Error_PM,rangos)
y0=opt["x"]
opt




######################  Experimentos #######################################
#y0=[-1.06405921e+00, -2.90823994e-07]# sol1
#y0=np.array([-4.46776363e+00, -2.83140995e-07])#sol 2
#y0=np.array([3.44343534e+00, 5.89490210e-07]) #sol 3
#y0=np.array([-1.78417868e+01, -2.95963256e-06])#sol 4
#y0=np.array([ 1.60679793e+01, -2.33867308e-06])#sol 5
y0=np.array([-3.91170708e+01,  2.34720559e-06]) #sol 6
#y0=np.array([ 3.80928831e+01, -1.21589543e-07])#sol 7


tt=np.arange(0,10*np.pi,.1)
sol = odeint(ecuacion_hamilton,y0 ,tt)
x=sol[:,0]
x_prima=psi(sol[:,1])
fig2,ax2 = plt.subplots()
ax2.plot(tt,x)
###############################################################################
############ Checking critical point type######################################
###############################################################################

####################Global Variables ######################################
w=2*np.pi/T
n=10000 #tamaño muestra puntos dentro de partición.
t=np.linspace(0,T,n+1)
sol = odeint(ecuacion_hamilton,y0 ,t)
x=sol[:,0]
x_prima=psi(sol[:,1])

################  Hessiano  ###########################
R=x**2 + 1
A=np.sin(R**gamma)
B=np.cos(R**gamma)
expr1=2*A*R**(2*gamma)*gamma**2*x**2 - 2*A*alpha**2*x**2\
    + 2*A*alpha*x**2 - 4*B*R**gamma*alpha*gamma*x**2\
        - 2*B*R**gamma*gamma**2*x**2 + 2*B*R**gamma*gamma*x**2\
            - R*(A*alpha + B*R**gamma*gamma)
a=1+2*np.abs(x_prima)
b=2*R**(alpha-2)*expr1


###############  Quadratic form ########################
n=20
M=np.zeros([2*n+1,2*n+1])

Phi=[np.ones_like(t)]
Phi_prima=[np.zeros_like(t)]

for j in range(1,n+1):
    Phi=Phi+[np.cos(j*w*t),np.sin(j*w*t)]
    Phi_prima=Phi_prima+[-np.sin(j*w*t)*j*w, np.cos(j*w*t)*j*w]
    
for j in range(2*n+1):
    for i in range(2*n+1):
        Integrando=a*Phi_prima[i]*Phi_prima[j]+b*Phi[i]*Phi[j]
        M[i,j]=simps(Integrando,x=t)

espectro=LA.eig(M)

print('autovalores=',espectro[0])

print('autovectores=',espectro[1])

fig, ax = plt.subplots(n+1,2,figsize=(10,30))

epsilon=np.arange(-1,1,.01)
I=np.zeros_like(epsilon)
            
        
            
for i in range(n+1): 
    for k in range(2):
        l=2*i+k
        V=espectro[1][:,l]
        
        eta=V[0]*np.ones_like(t)+sum(V[2*j+1]*np.cos((j+1)*w*t)+ 
                     V[2*(j+1)]*np.sin((j+1)*w*t) for j in range(n))
        eta_prima=V[0]*np.zeros_like(t)+sum(-V[2*j+1]*np.sin((j+1)*w*t)*(j+1)*w+ 
                     V[2*(j+1)]*np.cos((j+1)*w*t)*(j+1)*w for j in range(n))
        

        for h in range(len(epsilon)):
            u=x+epsilon[h]*eta
            u_prima=x_prima+epsilon[h]*eta_prima
            Integrando=u_prima**2/2.0+np.abs(u_prima)**3/3.0-\
                (1+u**2)**alpha*np.sin((1+u**2)**gamma)+u*np.cos(t)
            I[h]=simps(Integrando,x=t)
            
        ax[i,k].plot(epsilon,I)    
        ax[i,k].set_title('i='+str(i)+', '+'k='+str(k))









"""
s=ax.imshow(error, extent=[-20, 20, -5, 5], origin='lower',
           cmap='seismic')

fig.colorbar(s, ax=ax)
ax.set_aspect(aspect=3)



 




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X0,V0,error)
"""

