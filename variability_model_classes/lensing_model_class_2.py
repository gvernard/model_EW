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
class PrvModel():
    
    def __init__(self,cosmo,ra,dec,zs,M,zl,rv):
        self.cosmo = cosmo
        self.zs = zs
        self.zl = zl
        self.M  = M
        self.Dl = 1.0 # dummy value for initialization
        self.rv_arr = rv
        
        self.Nrv = len(self.rv_arr)
        self.dp_rv = np.empty(self.Nrv)
            
        # set the velocity model
        self.velObj = Velocity(ra,dec,zs,self.cosmo)
        print("Vcmb on the plane of the sky (unprojected) is: ",self.velObj.vcmb)
        # Calculate the Rice coeeficients without the time
        self.nu_rice,self.sigma_rice = self.getRiceCoeffs(self.M,self.zl)
        
    def getRiceCoeffs(self,mass,zl):
        Dl      = self.cosmo.angular_diameter_distance(zl).value
        self.Dl = Dl
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

    def Prv(self,t):
        nu_rice,sigma_rice = self.getRiceCoeffs(self.M,self.zl)

        for i in range(0,self.Nzl):
            dp_rv[i] = self.myrice(t*nu_rice,t*sigma_rice,self.rv_arr[i])

        return self.dp_rv
####################################################################################################




####################################################################################################
# Rs model
class PrsModel():
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
####################################################################################################



    
####################################################################################################
# Lensing magnification class
class MagModel():
    
    def __init__(self,rv,prv,mu0):
        r0 = self.radius(mu0)
        #x0  = r0
        #y0  = 0.0
        #self.priv_PrsModel = PrsModel(rv,prv,x0,y0)
        self.priv_PrsModel = PrsModel(rv,prv,r0)        
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
            fac = np.sqrt(self.mu_thres*self.mu_thres-1)
            A = fac/(self.mu_thres-fac)
            return A/(np.power(mu*mu-1,1.5))
####################################################################################################
