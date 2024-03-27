import sys
import numpy as np

import astropy.cosmology
import astropy.coordinates
import astropy.units as units

import matplotlib
import matplotlib.pyplot as plt

L_H = 4420 # This is the Hubble length (c/H0) in Mpc

def integrate(x_arr,y_arr):
    mysum = 0.0
    for i in range(0,len(x_arr)-1):
        mysum += (y_arr[i+1]+y_arr[i])*(x_arr[i+1]-x_arr[i])/2.0
    return mysum


def tau_int(zs):
    N = 50
    zz  = np.linspace(0.001,zs-0.001,N)
    
    integrand = np.zeros(N)
    for i in range(0,N):
        Dz = cosmo.angular_diameter_distance(zz[i]).value # in Mpc
        Dzz = cosmo.angular_diameter_distance_z1z2(zz[i],zs).value # in Mpc
        E = cosmo.efunc(zz[i]) # H(z) = efunc(z)*H0, we already factor out H0 
        integrand[i] = np.power(1+zz[i],2)*Dz*Dzz/E
        
    Dzs = cosmo.angular_diameter_distance(zs).value # in Mpc
    tau = integrate(zz,integrand)/Dzs
    return tau


def dev_tau_num(x,f):
    h = x[1]-x[0]
    dev = np.zeros(len(x))
    for i in range(1,len(x)-1):
        dev[i] = (-f[i-1]+f[i+1])/(2*h)
        
    dev[0] = (-1.5*f[0] + 2*f[1] - 0.5*f[2])/h
    dev[-1] = (0.5*f[-3] - 2*f[-2] + 1.5*f[-1])/h
    return dev



def dev_tau_ana(zs):
    N = 50
    zz  = np.linspace(0.001,zs-0.001,N)

    Ez  = cosmo.efunc(zs)
    Dcz = cosmo.angular_diameter_distance(zs).value*(1.0+zs)
    F1 = L_H/(Ez*Dcz)-1.0/(1.0+zs)
    F2 = 1.0/(Dcz*(1.0+zs))

    integrand1 = np.zeros(N)
    integrand2 = np.zeros(N)
    for i in range(0,N):
        Dczz = cosmo.angular_diameter_distance(zz[i]).value*(1.0+zz[i])
        Ezz  = cosmo.efunc(zz[i])
        fac = (1+zz[i])*Dczz/Ezz
        integrand1[i] = fac
        integrand2[i] = fac*Dczz
    I1 = integrate(zz,integrand1)
    I2 = integrate(zz,integrand2)

    tau = tau_int(zs)
    return -F1*tau + F1*I1 + F2*I2










Om0 = 0.3
cosmo = astropy.cosmology.LambdaCDM(H0=70,Om0=Om0,Ode0=0.7)
Odm = 0.85*Om0
A = 6.0*Odm/L_H


z = np.linspace(0,4,num=50)



fc1 = 1.0
fc2 = 0.1
taus_1 = [0]
taus_2 = [0]
dev_ana_1 = [0]
dev_ana_2 = [0]
for i in range(1,len(z)):
    tau_tmp = tau_int(z[i])
    taus_1.append( tau_tmp*fc1*A )
    taus_2.append( tau_tmp*fc2*A )

    dev_tau = dev_tau_ana(z[i])
    dev_ana_1.append( dev_tau*fc1*A )
    dev_ana_2.append( dev_tau*fc2*A )


dev_num_1 = dev_tau_num(z,taus_1)
dev_num_2 = dev_tau_num(z,taus_2)

print(len(taus_1),len(dev_num_1),len(dev_ana_1))

    
    
fig,ax = plt.subplots(figsize=(9.7,6),nrows=1,ncols=1)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax.plot(z,taus_1,label=r'$\tau$',color='black',ls='-')
#solids[1], = ax.plot(zs,P1,label=r'$P_1$',color=cycle[1])
#solids[2], = ax.plot(zs,Pmore,label=r'$P_{>1}$',color=cycle[2])

ax.plot(z,dev_num_1,label=r'$\tau$',color='black',ls='--')
#dashes[1], = ax.plot(zs,P1_2,label=r'$P_1$',color=cycle[1],ls='--')
#dashes[2], = ax.plot(zs,Pmore_2,label=r'$P_{>1}$',color=cycle[2],ls='--')

ax.plot(z,dev_ana_1,label=r'$\tau$',color='black',ls=':')


# Cosmetics
ax.set(xlabel=r'$z$',ylabel=r'dP')
ax.set_ylim(bottom=0,top=1.6)
ax.set_xlim(left=0,right=z[-1])
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))


fig.savefig("tau.pdf")

    
