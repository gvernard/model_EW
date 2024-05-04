import sys
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


Om0 = 0.3
cosmo = astropy.cosmology.LambdaCDM(H0=70,Om0=Om0,Ode0=0.7)
Odm = 0.85*Om0

fc = 1.0
M  = 1.0
zss = np.linspace(0.1,2,100)



n = np.empty(len(zss))
for j in range(0,len(zss)):
    zs = zss[j]
    zz = np.linspace(0.001,zs-0.001,100)
    Ds = cosmo.angular_diameter_distance(zs).value # in Mpc

    integrant = np.empty(len(zz))
    for i in range(0,len(zz)):
        Dzz = cosmo.angular_diameter_distance(zz[i]).value # in Mpc
        E = cosmo.efunc(zz[i])
        integrant[i] = np.power(Dzz/Ds,2)*np.power(1+zz[i],2)/E

    n[j] = integrate(zz,integrant)*156*fc/M


        
fig,ax = plt.subplots(figsize=(9.7,6),nrows=1,ncols=1)

ax.plot(zss,n,color='black',ls='-')

# Cosmetics
ax.set(xlabel=r'$z$',ylabel=r'$n$')
#ax.set_ylim(bottom=0,top=1.6)
#ax.set_xlim(left=0,right=z[-1])
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))


fig.savefig("nz.pdf")


    
