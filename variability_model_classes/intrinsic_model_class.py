import numpy as np
import math


class InModel_simple():
    sf_coeffs  = [-0.51,-0.479,0.113,0.18,0]
    tau_coeffs = [2.4,0.17,0.03,0.21,0]

    def __init__(self,lrest,Mi,zs,mu0,t):
        self.lrest = lrest # must be in Angstrom
        self.Mi = Mi
        self.zs = zs
        self.mu_0 = mu0
        self.t = t
        self.MBH_mean = 2.00 - 0.27*self.Mi
        self.MBH_sdev = 0.58 + 0.011*self.Mi

    def get_tau(self,log10_MBH):
        log_tau = self.tau_coeffs[0] + self.tau_coeffs[1]*np.log10(self.lrest/4000.0) + self.tau_coeffs[2]*(self.Mi+23) + self.tau_coeffs[3]*(log10_MBH-9.0) + self.tau_coeffs[4]*np.log10(1.0+self.zs)
        return np.power(10.0,log_tau)

    def get_sf(self,log10_MBH):
        log_sf = self.sf_coeffs[0] + self.sf_coeffs[1]*np.log10(self.lrest/4000.0) + self.sf_coeffs[2]*(self.Mi+23) + self.sf_coeffs[3]*(log10_MBH-9.0) + self.sf_coeffs[4]*np.log10(1.0+self.zs)
        return np.power(10.0,log_sf)

    def get_DRW_mean(self,tau):
        mean = np.exp(-self.t/tau)*np.log10(self.mu_0)
        return mean

    def get_DRW_sdev(self,SF,tau):
        sdev = 0.4*SF*np.sqrt(1.0-np.exp(-2*self.t/tau))/np.sqrt(2.0)
        return sdev

    def p_mu_noBH(self,mu):
        logMBH = self.MBH_mean - 0.4 # I can choose any fixed mass here
        tau = self.get_tau(logMBH)
        SF  = self.get_sf(logMBH)
        mean = self.get_DRW_mean(tau)
        sdev = self.get_DRW_sdev(SF,tau)
        return self.gaussian(np.log10(mu),mean,sdev)/(np.log(10.0)*mu)
    
    def p_x(self,x):
        logMBHs = np.linspace(self.MBH_mean-10*self.MBH_sdev,self.MBH_mean+10*self.MBH_sdev,100)
        dps = np.zeros(len(logMBHs))
        for i in range(0,len(logMBHs)):
            tau = self.get_tau(logMBHs[i])
            SF  = self.get_sf(logMBHs[i])
            mean = self.get_DRW_mean(tau)
            sdev = self.get_DRW_sdev(SF,tau)
            dps[i] = self.gaussian(x,mean,sdev)*self.gaussian(logMBHs[i],self.MBH_mean,self.MBH_sdev)
        return self.check_norm(logMBHs,dps)
    
    def p_mu(self,mu):
        return self.p_x(np.log10(mu))/(np.log(10.0)*mu)

    
    def p_x_mu0(self,x):
        logMBHs = np.linspace(self.MBH_mean-10*self.MBH_sdev,self.MBH_mean+10*self.MBH_sdev,100)
        dps = np.zeros(len(logMBHs))
        for i in range(0,len(logMBHs)):
            SF  = self.get_sf(logMBHs[i])
            sdev = 0.4*SF/np.sqrt(2.0)
            dps[i] = self.gaussian(x,0.0,sdev)*self.gaussian(logMBHs[i],self.MBH_mean,self.MBH_sdev)
        return self.check_norm(logMBHs,dps)
        
    def p_mu0(self,mu0):
        return self.p_x_mu0(np.log10(mu0))/(np.log(10.0)*mu0)

    def gaussian(self,x,mean,sdev):
        return 1.0/(sdev*math.sqrt(2.0*math.pi))*np.exp(-0.5*np.power((x-mean)/sdev,2.0))

    def check_norm(self,arr,dp):
        mynorm = 0.0
        for j in range(0,len(arr)-1):
            mynorm += (dp[j+1]+dp[j])*(arr[j+1]-arr[j])/2.0
        return mynorm
