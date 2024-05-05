import sys
import math
import numpy as np

import astropy.cosmology
import astropy.coordinates
import astropy.units as units

import matplotlib
import matplotlib.pyplot as plt


def integrate(x_arr,y_arr):
    mysum = 0.0
    for i in range(0,len(x_arr)-1):
        mysum += (y_arr[i+1]+y_arr[i])*(x_arr[i+1]-x_arr[i])/2.0
    return mysum


def optical_depth(cosmo,zs,fc,A):
    Ds = cosmo.angular_diameter_distance(zs).value # in Mpc

    zz = np.linspace(0.001,zs-0.001,100)
    integrant = np.empty(len(zz))
    for i in range(0,len(zz)):
        Dls = cosmo.angular_diameter_distance_z1z2(zz[i],zs).value # in Mpc
        Dl = cosmo.angular_diameter_distance(zz[i]).value # in Mpc
        E = cosmo.efunc(zz[i])
        integrant[i] = np.power(1+zz[i],2)*Dl*Dls/E

    tau = A*integrate(zz,integrant)/Ds
    return tau



def Pzlens(tau,k):
    tau = np.asarray(tau)
    scalar_input = False
    if tau.ndim == 0:
        tau = tau[None]  # Makes x 1D
        scalar_input = True

    prob = np.empty(len(tau))
    fac = math.factorial(k)
    for i in range(0,len(tau)):
        prob[i] = np.power(tau[i],k)*np.exp(-tau[i])/fac

    if scalar_input:
        return np.squeeze(prob)
    return prob




Om0 = 0.3       # Omega matter
Odm = 0.85*0.3  # Omega Dark Matter
cosmo = astropy.cosmology.LambdaCDM(H0=70,Om0=Om0,Ode0=0.7)
L_H = 4420 # This is the Hubble length (c/H0) in Mpc


fc = 1.0
zz = np.linspace(0.001,2,100)


A = 6.0*fc*Odm/L_H
tau = np.empty(len(zz))
for j in range(0,len(zz)):

    tau[j] = optical_depth(cosmo,zz[j],fc,A)




fig,ax = plt.subplots(figsize=(9.7,6),nrows=1,ncols=1)

ax.plot(zz,tau,color='black',ls='-')
ax.plot(zz,Pzlens(tau,0),color='black',ls='--')
ax.plot(zz,Pzlens(tau,1),color='black',ls=':')

# Cosmetics
ax.set(xlabel=r'$z$',ylabel=r'$P$')
#ax.set_ylim(bottom=0,top=1.6)
#ax.set_xlim(left=0,right=z[-1])
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))


fig.savefig("tau.pdf")


    
