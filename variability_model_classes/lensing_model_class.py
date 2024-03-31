import numpy as np
import math

import astropy.cosmology
import astropy.coordinates
import astropy.units as units

import scipy.stats


def equi_log(m_min,m_max,N):
    logmin  = np.log(m_min)
    logmax  = np.log(m_max)
    logvals = np.linspace(logmin,logmax,N)
    m_arr   = np.exp(logvals)
    return m_arr


def integrate(x_arr,y_arr):
    mysum = 0.0
    for i in range(0,len(x_arr)-1):
        mysum += (y_arr[i+1]+y_arr[i])*(x_arr[i+1]-x_arr[i])/2.0
    return mysum


def perform_comb_int(mu_co,mu_le,p_le,mu_in,p_in):
    mu_tmp  = np.concatenate( (equi_log(0.01,5,500),np.linspace(5.001,30,50)) ,axis=0)
    term_le = np.interp(mu_tmp,mu_le,p_le,right=0.0,left=0.0)

    pmu_co  = np.zeros(len(mu_co))
    for q in range(0,len(mu_co)):
        z = mu_co[q]/mu_tmp
        term_in = np.interp(z,mu_in,p_in,right=0.0,left=0.0)
        tmp = np.zeros(len(mu_tmp))
        for j in range(0,len(mu_tmp)):
            tmp[j] = term_le[j]*term_in[j]/mu_tmp[j]
        pmu_co[q] = integrate(mu_tmp,tmp)
        
    return pmu_co



####################################################################################################
# Different mass functions
class PowerLaw:
    norm = 1.0
    beta = 1.0

    def __init__(self,m_min,m_max,beta):
        self.beta = beta
        self.norm = (np.power(m_max,beta+1) - np.power(m_min,beta+1))/(beta+1)

    def p(self,m):
        return np.power(m,self.beta)/self.norm

    
class Kroupa:
    a_low  = 0.3
    a_mid  = 1.3
    a_high = 2.3
    m_low  = 0.08
    m_high = 0.5
    norm = 1.0

    def __init__(self,m_min,m_max):
        masses = equi_log(m_min,m_max,500)
        #masses = np.linspace(m_min,m_max,100)
        masses = equi_log(m_min,m_max,500)
        dum_p = np.empty(len(masses))
        for i in range(0,len(masses)):
            dum_p[i] = self.p(masses[i])
        mysum = 0.0
        for i in range(0,len(masses)-1):
            mysum += (dum_p[i+1]+dum_p[i])*(masses[i+1]-masses[i])/2.0
        self.norm = mysum
        
    def p(self,m):
        if m < self.m_low:
            val = np.power(m,-self.a_low)*np.power(self.m_high,-self.a_high+self.a_mid)*np.power(self.m_low,-self.a_mid+self.a_low)
        elif self.m_low <= m and m <= self.m_high:
            val = np.power(m,-self.a_mid)*np.power(self.m_high,-self.a_high+self.a_mid)
        else:
            val = np.power(m,-self.a_high)
        return val/self.norm
                

class Chabrier():
    fac = 0.158/(np.log(10.0))
    mean = np.log10(0.08)
    sdev = 0.69
    a = 2.3
    m_lim = 1
    norm = 1.0
    
    def __init__(self,m_min,m_max):
        #masses = np.linspace(m_min,m_max,100)
        masses = equi_log(m_min,m_max,500)
        dum_p = np.empty(len(masses))
        for i in range(0,len(masses)):
            dum_p[i] = self.p(masses[i])
        mysum = 0.0
        for i in range(0,len(masses)-1):
            mysum += (dum_p[i+1]+dum_p[i])*(masses[i+1]-masses[i])/2.0
        self.norm = mysum

    def p(self,m):
        if m < self.m_lim:
            val = (self.fac/m)*np.exp( -np.power(np.log10(m)-self.mean,2)/(2*self.sdev*self.sdev) )
        else:
            k = np.power(self.m_lim,self.a)*(self.fac/self.m_lim)*np.exp( -np.power(np.log10(self.m_lim)-self.mean,2)/(2*self.sdev*self.sdev) )
            val = k*np.power(m,-self.a)
        return val/self.norm


class Logarithmic():
    norm = 1.0
    def __init__(self,m_min,m_max):
        self.norm = (np.log(m_max)-np.log(m_min))
    def p(self,m):
        return (1.0/m)/self.norm
####################################################################################################


####################################################################################################
# Optical depth
class probZl:
    cosmo = astropy.cosmology.LambdaCDM(H0=70,Om0=0.3,Ode0=0.7)
    zs = 0.0
    tau_norm = 1.0
    L_H = 4420 # This is the Hubble length (c/H0) in Mpc

    def __init__(self,zs,cosmo):
        self.cosmo = cosmo
        self.zs = zs
        mysum = 0.0
        zz  = np.linspace(0.001,self.zs-0.001,100)
        for i in range(0,99):
            tau1 = self.optical_depth(zz[i])
            tau2 = self.optical_depth(zz[i+1])
            mysum += (tau1*np.exp(-tau1) + tau2*np.exp(-tau2))*(zz[i+1]-zz[i])/2.0
        self.tau_norm = mysum
            
    def optical_depth_integrand(self,zt_l,zt_s):
        Dol = self.cosmo.angular_diameter_distance(zt_l).value # in Mpc
        Dls = self.cosmo.angular_diameter_distance_z1z2(zt_l,zt_s).value # in Mpc
        dprob = np.power(1.0+zt_l,2.0)*(Dol)*(Dls)/self.cosmo.efunc(zt_l) # H(z) = efunc(z)*H0, we already factor out H0 
        return dprob

    def optical_depth(self,zs):
        #zt_l_arr = np.arange(0.01,zt_s,0.01)
        zz = np.linspace(0.001,zs,100)
        dprob = np.empty(len(zz))
        for i in range(0,len(zz)):
            dprob[i] = self.optical_depth_integrand(zz[i],zs)
        myint = integrate(zz,dprob)
        Ds = self.cosmo.angular_diameter_distance(zs).value # in Mpc
        return myint/(Ds) # Still need to multiply by the density parameter

    def dev_optical_depth(self,zs):
        N = 50
        zz  = np.linspace(0.001,zs-0.001,N)

        Ez  = self.cosmo.efunc(zs)
        Dcz = self.cosmo.angular_diameter_distance(zs).value*(1.0+zs)
        F1 = self.L_H/(Ez*Dcz)-1.0/(1.0+zs)
        F2 = 1.0/(Dcz*(1.0+zs))

        integrand1 = np.zeros(N)
        integrand2 = np.zeros(N)
        for i in range(0,N):
            Dczz = self.cosmo.angular_diameter_distance(zz[i]).value*(1.0+zz[i])
            Ezz  = self.cosmo.efunc(zz[i])
            fac = (1+zz[i])*Dczz/Ezz
            integrand1[i] = fac
            integrand2[i] = fac*Dczz
        I1 = integrate(zz,integrand1)
        I2 = integrate(zz,integrand2)

        tau = self.optical_depth(zs)
        return -F1*tau + F1*I1 + F2*I2

    def Pzlens(self,k,z,A):
        tau_z = A*self.optical_depth(z)
        dev_tau_z = A*self.dev_optical_depth(z)
        prob = np.power(tau_z,k)*np.exp(-tau_z)*np.abs(dev_tau_z)/np.math.factorial(k)
        #prob = np.power(tau_z,k)*np.exp(-tau_z)/np.math.factorial(k)
        return prob

    def PzAllLenses(self,z,A):
        tau_z = A*self.optical_depth(z)
        dev_tau_z = A*self.dev_optical_depth(z)
        prob = (1.0 - np.exp(-tau_z))*np.abs(dev_tau_z)
        return prob
        
####################################################################################################



####################################################################################################
# Velocity model
class Velocity:
    cosmo = astropy.cosmology.LambdaCDM(H0=70,Om0=0.3,Ode0=0.7)
    Ds  = 1.0
    zs  = 1.0    
    
    # CMB dipole magnitude and direction
    Vapex = 387.0 # km/s
    bapex = np.radians(90-48.26)
    lapex = np.radians(264.14-180)
    vcmb  = 0.0 # km/s
    
    spec_0 = 235 # km/s
    f0 = np.power(cosmo.Om0,0.6) + (1.0+cosmo.Om0/2.0)*cosmo.Ode0/70.0
    spec_s = 1.0
    spec_source_term = 1.0
    
    def __init__(self,ra,dec,zs,cosmo):
        self.cosmo = astropy.cosmology.LambdaCDM(H0=70,Om0=0.3,Ode0=0.7)
        self.zs = zs
        self.Ds = self.cosmo.angular_diameter_distance(zs).value # in Mpc
        self.spec_s = self.sigma_pec(zs)
        self.spec_source_term = np.power(self.spec_s/(1.0+self.zs),2)
        eq2ga = astropy.coordinates.SkyCoord(ra=ra,dec=dec,unit='degree').galactic
        b, l  = np.radians(np.array(eq2ga.b*units.deg)), np.radians(np.array(eq2ga.l*units.deg))
        self.vcmb = self.vcmb_obs(l,b)

    def vcmb_obs(self,l,b):
        # l and b must be galactic coordinates in radians
        # I need to do l-pi and pi/2-b to convert them to spherical coordinates.
        vcmb = np.array([self.Vapex*np.cos(self.lapex)*np.sin(self.bapex),self.Vapex*np.sin(self.lapex)*np.sin(self.bapex),self.Vapex*np.cos(self.bapex)])
        cor = np.array([np.cos(l-math.pi)*np.sin(math.pi/2.0-b),np.sin(l-math.pi)*np.sin(math.pi/2.0-b),np.cos(math.pi/2.0-b)])
        cor = cor/(cor[0]**2+cor[1]**2+cor[2]**2)**0.5
        alpha = vcmb.dot(cor)
        alpha2 = alpha*cor
        r = -alpha2.T + vcmb
        norm = np.linalg.norm(r)
        return norm

    def vcmb_proj(self,zl,Dl,Dls):
        vcmb_proj = (Dls/Dl)*self.vcmb/(1.0+zl)
        return vcmb_proj

    def sigma_pec(self,z):
        fz = np.power(self.cosmo.Om(z),0.6)
        spec_z = (fz/self.f0)*self.spec_0/np.sqrt(1.0+z)
        return spec_z

    def sigma_v(self,zl,Dl):
        spec_l = self.sigma_pec(zl)
        sigma_v2 = np.power( (spec_l/(1.0+zl))*(self.Ds/Dl) ,2) + self.spec_source_term
        return np.sqrt(sigma_v2)
####################################################################################################




####################################################################################################
# Rv model with fixed mass
class PrvModel_fixed_mass_nonorm():
    Nzl    = 300
    zl_arr = np.empty(Nzl)
    pzl    = np.empty(Nzl)
    fc = 1.0
    Odm = 0.85*0.3 # 85% of the matter density Omega_M
    L_H = 4420 # This is the Hubble length (c/H0) in Mpc
    
    def __init__(self,cosmo,ra,dec,zs,fc,rvs=None):
        self.cosmo = cosmo
        self.zs = zs
        self.fc = fc
        
        # set the lens redshift probability
        self.A = 6.0*self.fc*self.Odm/self.L_H
        self.lensRedshiftPrior(zs)
            
        # rv output vector
        if rvs is None:
            self.rv_arr = np.concatenate((np.linspace(0.0,0.5,500),np.linspace(0.501,15,100)),axis=0)
        else:
            self.rv_arr = rvs
        self.Nrv = len(self.rv_arr)
        self.dp_rv = np.empty(self.Nrv)
            
        check = False
        if check:
            # load the rice coefficients without the time
            dum = 0
        else:
            # set the velocity model
            self.velObj = Velocity(ra,dec,zs,self.cosmo)
            print("Vcmb on the plane of the sky (unprojected) is: ",self.velObj.vcmb)
            # Calculate the Rice coeeficients without the time
            self.nu_rice,self.sigma_rice = self.getRiceCoeffs_M(1.0)

    def lensRedshiftPrior(self,zs):
        zl_min = 0.001
        zl_max = zs-0.001
        self.zl_arr = np.linspace(zl_min,zl_max,self.Nzl)
        myTauModel = probZl(zs,self.cosmo)
        for i in range(0,self.Nzl):
            #Dl  = self.cosmo.angular_diameter_distance(self.zl_arr[i]).value
            #tau = self.fc*self.Odm*tauP.optical_depth(self.zl_arr[i],Dl)
            #self.pzl[i] = tau*np.exp(-tau)
            #dev_tau = myTauModel.dev_optical_depth(zs)
            self.pzl[i] = myTauModel.PzAllLenses(zs,self.A)

    def getRiceCoeffs_M(self,mass):
        nu_rice    = np.empty(self.Nzl)
        sigma_rice = np.empty(self.Nzl)
        for i in range(0,self.Nzl):
            nu_rice[i],sigma_rice[i] = self.getRiceCoeffs(mass,self.zl_arr[i])
        print("Rice distribution nu,sigma coefficients ready for a given MASS")
        return nu_rice,sigma_rice

    def getRiceCoeffs(self,mass,zl):
        Dl      = self.cosmo.angular_diameter_distance(zl).value
        Dls     = self.cosmo.angular_diameter_distance_z1z2(zl,self.zs).value
        vcmb    = self.velObj.vcmb_proj(zl,Dl,Dls)
        sigma_v = self.velObj.sigma_v(zl,Dl)
        RE      = self.Rein(Dl,self.velObj.Ds,Dls,mass)
        nu_rice = 0.03154*vcmb/RE
        sigma_rice = 0.03154*sigma_v/RE
        return nu_rice,sigma_rice
    
    def Rein(self,Dl,Ds,Dls,M):
        return 13.5*np.sqrt(M*Dls*Ds/Dl) # in 10^14 cm
    
    def myrice(self,n,s,r):
        termA = r/np.power(s,2)
        termB = scipy.special.iv(0,r*n/np.power(s,2))
        termC = np.exp(-(np.power(r,2)+np.power(n,2))/(2*np.power(s,2)))
        # I need the following check because for very small values of n I get an overflow error (number too small to be represented as double)
        if termA == 0 or termB == 0 or termC == 0:
            return 0.0
        else:
            return termA*termB*termC

    def integral_over_zl_only(self,nu_rice,sigma_rice):
        # One dimensional integral on zl, for fixed mass
        for k in range(0,self.Nrv):
            dp = np.empty(self.Nzl)
            for i in range(0,self.Nzl):
                dp[i] = self.myrice(nu_rice[i],sigma_rice[i],self.rv_arr[k])*self.pzl[i]
            self.dp_rv[k] = integrate(self.zl_arr,dp)

    def Prv_M(self,t,mass):
        # Probability for a given M (integrated over zl)
        nu_rice,sigma_rice = self.getRiceCoeffs_M(mass)
        self.integral_over_zl_only(t*nu_rice,t*sigma_rice)
        return self.rv_arr,self.dp_rv
####################################################################################################





####################################################################################################
# Rv model with fixed mass
class PrvModel_fixed_mass():
    Nzl    = 300
    zl_arr = np.empty(Nzl)
    pzl    = np.empty(Nzl)
    fc = 1.0
    Odm = 0.85*0.3 # 85% of the matter density Omega_M
    
    
    def __init__(self,cosmo,ra,dec,zs,fc,rvs=None):
        self.cosmo = cosmo
        self.zs = zs
        self.fc = fc
        
        # set the lens redshift probability
        self.lensRedshiftPrior(zs)
        norm = integrate(self.zl_arr,self.pzl)
        print("Lens redshift prior normalization: ",norm)
        for i in range(0,len(self.zl_arr)):
            self.pzl[i] = self.pzl[i]/norm
            
        # rv output vector
        if rvs is None:
            self.rv_arr = np.concatenate((np.linspace(0.0,0.5,500),np.linspace(0.501,15,100)),axis=0)
        else:
            self.rv_arr = rvs
        self.Nrv = len(self.rv_arr)
        self.dp_rv = np.empty(self.Nrv)
            
        check = False
        if check:
            # load the rice coefficients without the time
            dum = 0
        else:
            # set the velocity model
            self.velObj = Velocity(ra,dec,zs,self.cosmo)
            print("Vcmb on the plane of the sky (unprojected) is: ",self.velObj.vcmb)
            # Calculate the Rice coeeficients without the time
            self.nu_rice,self.sigma_rice = self.getRiceCoeffs_M(1.0)


    def lensRedshiftPrior(self,zs):
        zl_min = 0.001
        zl_max = zs-0.001
        self.zl_arr = np.linspace(zl_min,zl_max,self.Nzl)
        tauP = probZl(zs,self.cosmo)
        for i in range(0,self.Nzl):
            Dl  = self.cosmo.angular_diameter_distance(self.zl_arr[i]).value
            tau = self.fc*self.Odm*tauP.optical_depth(self.zl_arr[i],Dl)
            self.pzl[i] = tau*np.exp(-tau)
            #self.pzl[i] = 1 - np.exp(-tau)

    def getRiceCoeffs_M(self,mass):
        nu_rice    = np.empty(self.Nzl)
        sigma_rice = np.empty(self.Nzl)
        for i in range(0,self.Nzl):
            nu_rice[i],sigma_rice[i] = self.getRiceCoeffs(mass,self.zl_arr[i])
        print("Rice distribution nu,sigma coefficients ready for a given MASS")
        return nu_rice,sigma_rice

    def getRiceCoeffs(self,mass,zl):
        Dl      = self.cosmo.angular_diameter_distance(zl).value
        Dls     = self.cosmo.angular_diameter_distance_z1z2(zl,self.zs).value
        vcmb    = self.velObj.vcmb_proj(zl,Dl,Dls)
        sigma_v = self.velObj.sigma_v(zl,Dl)
        RE      = self.Rein(Dl,self.velObj.Ds,Dls,mass)
        nu_rice = 0.03154*vcmb/RE
        sigma_rice = 0.03154*sigma_v/RE
        return nu_rice,sigma_rice
    
    def Rein(self,Dl,Ds,Dls,M):
        return 13.5*np.sqrt(M*Dls*Ds/Dl) # in 10^14 cm
    
    def myrice(self,n,s,r):
        termA = r/np.power(s,2)
        termB = scipy.special.iv(0,r*n/np.power(s,2))
        termC = np.exp(-(np.power(r,2)+np.power(n,2))/(2*np.power(s,2)))
        # I need the following check because for very small values of n I get an overflow error (number too small to be represented as double)
        if termA == 0 or termB == 0 or termC == 0:
            return 0.0
        else:
            return termA*termB*termC

    def integral_over_zl_only(self,nu_rice,sigma_rice):
        # One dimensional integral on zl, for fixed mass
        for k in range(0,self.Nrv):
            dp = np.empty(self.Nzl)
            for i in range(0,self.Nzl):
                dp[i] = self.myrice(nu_rice[i],sigma_rice[i],self.rv_arr[k])*self.pzl[i]
            self.dp_rv[k] = integrate(self.zl_arr,dp)

    def Prv_M(self,t,mass):
        # Probability for a given M (integrated over zl)
        nu_rice,sigma_rice = self.getRiceCoeffs_M(mass)
        self.integral_over_zl_only(t*nu_rice,t*sigma_rice)
        return self.rv_arr,self.dp_rv
####################################################################################################




####################################################################################################
# Rv model
class PrvModel():
    # Mass prior parameters
    Nm    = 80
    pm    = np.empty(Nm)

    # Lens redshift prior parameters
    Nzl    = 300
    zl_arr = np.empty(Nzl)
    pzl    = np.empty(Nzl)


    def __init__(self,cosmo,ra,dec,zs,imf_type,m_min=0.01,m_max=100,rvs=None):
        self.cosmo = cosmo
        self.zs = zs

        # Set the mass probability
        self.m_min = m_min
        self.m_max = m_max
        self.m_arr = equi_log(self.m_min,self.m_max,self.Nm)
        self.massPrior(imf_type)
        print("Mass prior normalization: ",integrate(self.m_arr,self.pm))

        # set the lens redshift probability
        self.lensRedshiftPrior(zs)
        print("Lens redshift prior normalization: ",integrate(self.zl_arr,self.pzl))

        # rv output vector
        if rvs is None:
            self.rv_arr = np.concatenate((np.linspace(0.0,0.5,500),np.linspace(0.501,15,100)),axis=0)
        else:
            self.rv_arr = rvs
        self.Nrv = len(self.rv_arr)
        self.dp_rv = np.empty(self.Nrv)
            
        
        check = False
        if check:
            # load the rice coefficients without the time
            dum = 0
        else:
            # set the velocity model
            self.velObj = Velocity(ra,dec,zs,self.cosmo)
            print("Vcmb on the plane of the sky (unprojected) is: ",self.velObj.vcmb)
            # Calculate the Rice coeeficients without the time
            self.nu_rice,self.sigma_rice = self.getRiceCoeffs_MZ()
        
    def lensRedshiftPrior(self,zs):
        zl_min = 0.001
        zl_max = zs-0.001
        self.zl_arr = np.linspace(zl_min,zl_max,self.Nzl)
        tauP = probZl(zs,self.cosmo)
        for i in range(0,self.Nzl):
            Dl  = self.cosmo.angular_diameter_distance(self.zl_arr[i]).value
            self.pzl[i] = tauP.Pzlens(self.zl_arr[i],Dl)

    def massPrior(self,imf_type):
        if imf_type == 'salpeter':
            prior_obj = PowerLaw(self.m_min,self.m_max,-2.35)
        elif imf_type == 'kroupa':
            prior_obj = Kroupa(self.m_min,self.m_max)
        elif imf_type == 'chabrier':
            prior_obj = Chabrier(self.m_min,self.m_max)
        elif imf_type == 'logarithmic':
            prior_obj = Logarithmic(self.m_min,self.m_max)
        else:
            prior_obj = PowerLaw(self.m_min,self.m_max,0) # A uniform prior
        for j in range(0,self.Nm):
            self.pm[j] = prior_obj.p(self.m_arr[j])

    def getRiceCoeffs_MZ(self):
        nu_rice    = np.empty((self.Nzl,self.Nm))
        sigma_rice = np.empty((self.Nzl,self.Nm))
        for i in range(0,self.Nzl):
            zl  = self.zl_arr[i]
            Dl  = self.cosmo.angular_diameter_distance(zl).value
            Dls = self.cosmo.angular_diameter_distance_z1z2(zl,self.zs).value
            vcmb = self.velObj.vcmb_proj(zl,Dl,Dls)
            sigma_v = self.velObj.sigma_v(zl,Dl)

            for j in range(0,self.Nm):
                m  = self.m_arr[j]
                RE = self.Rein(Dl,self.velObj.Ds,Dls,m)
                nu_rice[i,j]    = 0.03154*vcmb/RE
                sigma_rice[i,j] = 0.03154*sigma_v/RE
                #print(sigma_rice[i,j]/nu_rice[i,j])
        print("Rice distribution nu,sigma coefficients ready")
        return nu_rice,sigma_rice

    def getRiceCoeffs_Z(self,mass):
        nu_rice    = np.empty(self.Nzl)
        sigma_rice = np.empty(self.Nzl)
        for i in range(0,self.Nzl):
            nu_rice[i],sigma_rice[i] = self.getRiceCoeffs(mass,self.zl_arr[i])
        print("Rice distribution nu,sigma coefficients ready for a given MASS")
        return nu_rice,sigma_rice

    def getRiceCoeffs(self,mass,zl):
        Dl      = self.cosmo.angular_diameter_distance(zl).value
        Dls     = self.cosmo.angular_diameter_distance_z1z2(zl,self.zs).value
        vcmb    = self.velObj.vcmb_proj(zl,Dl,Dls)
        sigma_v = self.velObj.sigma_v(zl,Dl)
        RE      = self.Rein(Dl,self.velObj.Ds,Dls,mass)
        nu_rice = 0.03154*vcmb/RE
        sigma_rice = 0.03154*sigma_v/RE
        return nu_rice,sigma_rice
    
    def Rein(self,Dl,Ds,Dls,M):
        return 13.5*np.sqrt(M*Dls*Ds/Dl) # in 10^14 cm
    
    def myrice(self,n,s,r):
        termA = r/np.power(s,2)
        termB = scipy.special.iv(0,r*n/np.power(s,2))
        termC = np.exp(-(np.power(r,2)+np.power(n,2))/(2*np.power(s,2)))
        # I need the following check because for very small values of n I get an overflow error (number too small to be represented as double)
        if termA == 0 or termB == 0 or termC == 0:
            return 0.0
        else:
            return termA*termB*termC

        
    def integral_over_zl_and_m(self,nu_rice,sigma_rice):
        # Two dimensional integral on zl and m
        for k in range(0,self.Nrv):

            dp = np.empty((self.Nzl,self.Nm))
            for i in range(0,self.Nzl):
                for j in range(0,self.Nm):
                    dp[i,j] = self.myrice(nu_rice[i,j],sigma_rice[i,j],self.rv_arr[k])*self.pzl[i]*self.pm[j]

            mysum_over_m = np.empty(self.Nzl)
            for i in range(0,self.Nzl-1):
                mysum = 0.0
                for j in range(0,self.Nm-1):
                    mysum += (dp[i,j+1] + dp[i,j])*(self.m_arr[j+1] - self.m_arr[j])/2.0
                mysum_over_m[i] = mysum

            mysum = 0.0
            for i in range(0,self.Nzl-1):    
                mysum += (mysum_over_m[i+1] + mysum_over_m[i])*(self.zl_arr[i+1] - self.zl_arr[i])/2.0        
            self.dp_rv[k] = mysum

            
    def Prv_MZ(self,t):
        # Probability integrated over M and zl
        self.integral_over_zl_and_m(t*self.nu_rice,t*self.sigma_rice)
        return self.rv_arr,self.dp_rv

    
    def integral_over_zl_only(self,nu_rice,sigma_rice):
        # One dimensional integral on zl, for fixed mass
        for k in range(0,self.Nrv):
            dp = np.empty(self.Nzl)
            for i in range(0,self.Nzl):
                dp[i] = self.myrice(nu_rice[i],sigma_rice[i],self.rv_arr[k])*self.pzl[i]
            mysum = 0.0
            for i in range(0,self.Nzl-1):
                mysum += (dp[i+1] + dp[i])*(self.zl_arr[i+1] - self.zl_arr[i])/2.0
            self.dp_rv[k] = mysum

    def Prv_Z(self,t,mass):
        # Probability for a given M (integrated over zl)
        nu_rice,sigma_rice = self.getRiceCoeffs_Z(mass)
        self.integral_over_zl_only(t*nu_rice,t*sigma_rice)
        return self.rv_arr,self.dp_rv

    
    def Prv(self,t,mass,zl):
        # Probability for a given M and zl
        nu_rice,sigma_rice = self.getRiceCoeffs(mass,zl)
        for k in range(0,self.Nrv):
            self.dp_rv[k] = self.myrice(t*nu_rice,t*sigma_rice,self.rv_arr[k])
        return self.rv_arr,self.dp_rv    
####################################################################################################


    
####################################################################################################
# Rs model
class PrsModel2():
    def __init__(self,rv,prv,r0):
        self.rv = rv
        self.prv = prv
        self.r0 = r0

    def Prv(self,r):
        prv = np.interp(r,self.rv,self.prv,left=0.0,right=0.0)
        return prv
        
    def P(self,rs):
        dd = 2*rs/4000.0
        int_rv = np.linspace(abs(self.r0-rs)+dd,self.r0+rs-dd,400)
        #int_rv = np.linspace(abs(self.r0-rs),self.r0+rs,4000,endpoint=False)[1:]
        dps = np.zeros(len(int_rv))
        for i in range(0,len(int_rv)):
            A = np.power(int_rv[i],2) - np.power(self.r0-rs,2)
            B = np.power(self.r0+rs,2) - np.power(int_rv[i],2)
            #print(int_rv[i],self.Prv(int_rv[i]),rs,A,B)
            dps[i] = self.Prv(int_rv[i])/np.sqrt(A*B)
        term = 0.0
        for i in range(0,len(int_rv)-1):
            term += (dps[i]+dps[i+1])*(int_rv[i+1]-int_rv[i])/2.0
        return 2.0*rs*term/math.pi

    

class PrsModel():
    x0 = 1.0
    y0 = 1.0

    def __init__(self,rv,prv,x0,y0):
        self.x0 = x0
        self.y0 = y0
        self.rv = rv
        self.prv = prv
        
    def Prv(self,r):
        prv = np.interp(r,self.rv,self.prv,left=0.0,right=0.0)
        return prv
        
    def H1(self,rs,ys):
        return np.sqrt(rs*rs - ys*ys)

    def H2(self,rs,ys):
        return -np.sqrt(rs*rs - ys*ys)

    def P(self,rs):
        # I need to perform the integral of d_ys from -rs to rs.
        # But because ys appears always squared, I can do it from 0 to rs and multiply by 2.
        if rs == 0:
            return 0

        ys = np.linspace(0,rs-0.0001,1000)
        #ys = np.linspace(0,rs/np.sqrt(2.0),500)
                      
        dps = np.zeros(len(ys))
        for i in range(0,len(ys)):
            uk     = self.H1(rs,ys[i])
            rv     = np.hypot(uk-self.x0,ys[i]-self.y0)
            dps[i] = self.Prv(rv)*abs(rs/uk)/(2*math.pi*rv)
        term1 = 0.0
        for i in range(0,len(ys)-1):
            term1 += (dps[i]+dps[i+1])*(ys[i+1]-ys[i])/2.0

        dps = np.zeros(len(ys))
        for i in range(0,len(ys)):
            uk     = self.H2(rs,ys[i])
            rv     = np.hypot(uk-self.x0,ys[i]-self.y0)
            dps[i] = self.Prv(rv)*abs(rs/uk)/(2*math.pi*rv)
        term2 = 0.0
        for i in range(0,len(ys)-1):
            term2 += (dps[i]+dps[i+1])*(ys[i+1]-ys[i])/2.0

        return 2*(term1+term2)

####################################################################################################






####################################################################################################
# Lensing magnification class
class MagModel():
    
    def __init__(self,rv,prv,mu0):
        r0 = self.radius(mu0)
        #x0  = r0
        #y0  = 0.0
        #self.priv_PrsModel = PrsModel(rv,prv,x0,y0)
        self.priv_PrsModel = PrsModel2(rv,prv,r0)        
        self.mu_thres = 1.061
        self.mu_0 = mu0

    def reset_mu0(self,mu0):
        self.mu_0 = mu0
        r0 = self.radius(mu0)
        self.priv_PrsModel.r0 = r0
        
    def radius(self,mu):
        r2 = 2*( mu/np.sqrt(mu*mu-1) - 1)
        return np.sqrt(r2)
    
    def p_mu(self,x):
        if x <= 1.0:
            return 0.0
        else:
            r    = self.radius(x)
            fac1 = self.priv_PrsModel.P(r)
            fac2 = 1.0/(r*np.power(x*x-1,1.5))
            A = 1#np.sqrt(self.mu_thres*self.mu_thres-1)/(self.mu_thres-np.sqrt(self.mu_thres*self.mu_thres-1))
            return A*fac1*fac2

    def p_mu0(self,mu):
        if mu < self.mu_thres:
            return 0.0
        else:
            A = np.sqrt(self.mu_thres*self.mu_thres-1)/(self.mu_thres-np.sqrt(self.mu_thres*self.mu_thres-1))
            return A/(np.power(mu*mu-1,1.5))
####################################################################################################
