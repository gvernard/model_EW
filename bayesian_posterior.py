import re
import os
import sys
import json
import numpy as np
import astropy.cosmology


sys.path.append(os.path.join(os.path.dirname(__file__), "variability_model_classes"))

from intrinsic_model_class import InModel_simple
from lensing_model_class import *


def gaussian_arr(mean,sdev):
    sig = 3
    low  = max(0,mean - sig*sdev)
    high = min(mean + sig*sdev,1.0)
    xarr = np.linspace(low,high,50)

    y = np.zeros(len(xarr))
    fac = np.sqrt(math.pi*2)
    for i in range(0,len(xarr)):
        if xarr[i] < low or xarr[i] > high:
            y[i] = 0.0
        else:
            gaussian(mean,sdev,xarr[i])
    return xarr,y

def gaussian(mean,sdev,x):
    fac = sdev*np.sqrt(math.pi*2)
    expo  = -0.5*np.power((mean-x)/sdev,2)
    return np.exp(expo)/fac
    

def equi_log10(min,max,N):
    logmin  = np.log10(min)
    logmax  = np.log10(max)
    logvals = np.linspace(logmin,logmax,N)
    arr   = np.power(10,logvals)
    return arr




qid = sys.argv[1]
input_path = os.path.join(sys.argv[2],'')
#intrinsic_var_file = "/home/giorgos/unbacked_up_data/sdss_quasar_spectra/"+myjson["id"]+"/chabrier/in_dps_ratios.dat"
input_file = os.path.join(input_path,qid,'input_fit.json')
intrinsic_var_file = os.path.join(input_path,qid,"in_dps_ratios.dat")

if len(sys.argv) > 3:
    output_path = sys.argv[3]
else:
    output_path = ''
output_file = os.path.join(output_path,qid+'_post.txt')
output_file_grid = os.path.join(output_path,qid+'_grid.txt')





# Get json file with input parameters
f          = open(input_file,'r')
input_str  = f.read()
input_str  = re.sub('//.*?\n|/\*.*?\*/','',input_str,flags=re.S)
myjson     = json.loads(input_str)

# for lensing
ra  = myjson['ra']
dec = myjson['dec']
zs  = myjson['zs']
cosmo = astropy.cosmology.LambdaCDM(H0=70,Om0=0.3,Ode0=0.7)

# for intrinsic
Mi    = myjson['Mi']
lrest = myjson['lrest']

# for both
t = myjson["t"] # in years

# Data
r_data = myjson["r"]
sigma  = myjson["sigma"]

fcs = myjson["fcs"]
masses = myjson["masses"]



print("PBH/DM density ratio: ",len(fcs))
print(fcs)
print("Masses: ",len(masses))
print(masses)




# Setting up other variables
le_ratios = np.concatenate( (np.linspace(0.1,0.75,40,endpoint=False),np.linspace(0.75,1.0,50,endpoint=False),np.linspace(1.0,1.1,10,endpoint=False),np.linspace(1.1,1.5,10)) ,axis=0)
#normal_x,normal_y = gaussian_arr(r_data,sigma)
#co_ratios = np.linspace(normal_x[0],normal_x[-1],50)
co_ratios = np.linspace(0.1,1.5,100)

# Creates a new file
with open(output_file,'w') as fp:
    pass
with open(output_file_grid,'w') as fp:
    pass



# Calculate ratios from the intrinsic model
##################################################################################
print("Intrinsic model...")
in_ratios,p_in_ratios = np.loadtxt(intrinsic_var_file,unpack=True)
myInModel = InModel_simple(lrest,Mi,zs,2.0,365.0*t) # mu0 is dummy here, time has to be in days, required for p_mu0 in computing the combined model below

'''
in_ratios = np.concatenate( (np.linspace(0.1,0.8,40,endpoint=False),np.linspace(0.8,1.2,50,endpoint=False),np.linspace(1.2,2,40)) ,axis=0)
myInModel = InModel_simple(lrest,Mi,zs,2.0,365.0*t) # mu0 is dummy here, time has to be in days

p_in_ratios = np.zeros(len(in_ratios))
mu0_arr     = np.linspace(0.01,20,200)
for i in range(0,len(in_ratios)):
    print("ratio ",round(in_ratios[i],4),'   ',i,"/",len(in_ratios))

    tmp = np.zeros(len(mu0_arr))
    for j in range(0,len(mu0_arr)):
        mu0 = mu0_arr[j]
        myInModel.mu_0 = mu0
        tmp[j] = mu0*myInModel.p_mu(in_ratios[i]*mu0)*myInModel.p_mu0(mu0)
    p_in_ratios[i] = integrate(mu0_arr,tmp)
print(integrate(in_ratios,p_in_ratios))
np.savetxt(path+"in_dps_ratios.dat",np.c_[in_ratios,p_in_ratios])
'''
print("done")



# The following loop parses first the fraction (descending) and then the mass (ascending).
# So it produces a fcs-mass (y-x) grid.
for fc in fcs:
    for mass in masses:


        # Calculate ratios from the lensing model
        ##################################################################################
        print("Lensing model...")
        #myPrvModel = PrvModel_fixed_mass(cosmo,ra,dec,zs,fc)
        myPrvModel = PrvModel_fixed_mass_nonorm(cosmo,ra,dec,zs,fc)
        rv,prv     = myPrvModel.Prv_M(t,mass)
        myMagModel = MagModel(rv,prv,2.0) # mu0 is dummy here
        
        p_le_ratios = np.zeros(len(le_ratios))
        for i in range(0,len(le_ratios)):
            #print("ratio ",round(le_ratios[i],4),'   ',i,"/",len(le_ratios))

            if 0.7 < le_ratios[i] and le_ratios[i] < 1.0:
                mu0_arr = np.concatenate( (equi_log(1.001/le_ratios[i],3.000/le_ratios[i],800),equi_log(3.001/le_ratios[i],100,200)) ,axis=0)
            else:
                #mu0_arr = np.concatenate( (np.linspace(1.001,3.001,200),np.linspace(3.002,100,100)) ,axis=0)
                mu0_arr = np.concatenate( (equi_log(1.001,3.001,200),equi_log(3.002,100,100)) ,axis=0)

            tmp = np.zeros(len(mu0_arr))
            for j in range(0,len(mu0_arr)):
                mu0 = mu0_arr[j]
                myMagModel.reset_mu0(mu0)
                tmp[j] = mu0*myMagModel.p_mu(le_ratios[i]*mu0)*myMagModel.p_mu0(mu0)
            p_le_ratios[i] = integrate(mu0_arr,tmp)
        print(integrate(le_ratios,p_le_ratios))
        #np.savetxt(path+"le_dps_ratios.dat",np.c_[le_ratios,p_le_ratios])
        print("done")




        # Calculate ratios from combining both models
        ##################################################################################
        print("Combined model...")
        if t<2:
            print("ATTENTION: the intrinsic part of the combined model is assumed to be the one for t->infinity, which is true for t >= 2 years. Here t=",t)
            sys.exit()


        ### Calculating Pmu0
        #===================
        # Caclulate pmu0 from the lensing model
        le_mu0  = np.concatenate( (equi_log(1.001,3,200),np.linspace(3.001,50,100)) ,axis=0)
        #le_mu0  = np.concatenate( (np.linspace(1.001,2,200),np.linspace(2.1,50,100)) ,axis=0)
        le_pmu0 = np.zeros(len(le_mu0))
        for i in range(0,len(le_mu0)):
            le_pmu0[i] = myMagModel.p_mu0(le_mu0[i])
        print(integrate(le_mu0,le_pmu0))
    
        # Caclulate pmu0 from the intrinsic variability model
        #in_mu0  = np.linspace(0.001,20,200)
        in_mu0  = np.concatenate( (equi_log(0.001,3,200),np.linspace(3.1,100,100)) ,axis=0)
        in_pmu0 = np.zeros(len(in_mu0))
        for i in range(0,len(in_mu0)):
            in_pmu0[i] = myInModel.p_mu0(in_mu0[i])
        print(integrate(in_mu0,in_pmu0))

        # Calculate pmu0 for the combined model interpolating the pmu0 from lensing and intrinsic
        #co_mu0  = np.linspace(0.001,20,400)
        co_mu0  = np.concatenate( (equi_log(0.001,5,200),np.linspace(5.001,30,50)) ,axis=0)
        co_pmu0 = perform_comb_int(co_mu0,le_mu0,le_pmu0,in_mu0,in_pmu0)
        print(integrate(co_mu0,co_pmu0))


        ### Calculating Pmu
        #==================
        mu0_arr = np.linspace(1.061,5,50)
        mu_arr = np.concatenate( (equi_log(1.001,5,200),np.linspace(5.001,100,100)) ,axis=0)

        # Step 2: get the pmu for the combined model for given mu0
        le_pmu_given_mu0 = []
        for i in range(0,len(mu0_arr)):
            myMagModel.reset_mu0(mu0_arr[i])
            tmp = np.zeros(len(mu_arr))
            for j in range(0,len(mu_arr)):
                tmp[j] = myMagModel.p_mu(mu_arr[j])
            le_pmu_given_mu0.append(tmp)

        co_mu  = np.linspace(0.001,20,400)
        co_pmu_given_mu0 = []
        for i in range(0,len(mu0_arr)):
            co_pmu = perform_comb_int(co_mu,mu_arr,le_pmu_given_mu0[i],in_mu0,in_pmu0)
            co_pmu_given_mu0.append(co_pmu)

        # Step 3: get the ratio for the combined model from co_pmu0 and co_pmu
        p_co_ratios = np.zeros(len(co_ratios))
        for i in range(0,len(co_ratios)):
            term_mu0 = np.interp(mu0_arr,co_mu0,co_pmu0,right=0.0,left=0.0)
            tmp = np.zeros(len(mu0_arr))
            for j in range(0,len(mu0_arr)):
                tmp[j] = mu0_arr[j]*np.interp(co_ratios[i]*mu0_arr[j],co_mu,co_pmu_given_mu0[j],right=0.0,left=0.0)*term_mu0[j]
            p_co_ratios[i] = integrate(mu0_arr,tmp)
        
        print(integrate(co_ratios,p_co_ratios))
        #np.savetxt(path+"co_dps_ratios.dat",np.c_[co_ratios,p_co_ratios])
        print("done")




        # Perform final integral
        ##################################################################################
        integrand = np.zeros(len(co_ratios))
        for i in range(0,len(co_ratios)):
            integrand[i] = gaussian(r_data,sigma,co_ratios[i])*p_co_ratios[i]
            #integrand[i] = normal_y[i]*p_co_ratios[i]
        post = integrate(co_ratios,integrand)
        print(mass,fc,post)

        
        with open(output_file,'a') as myfile:
            myfile.write(f'{mass:.8f},{fc:.8f},{post:.10f}\n')

        with open(output_file_grid,'a') as myfile:
            myfile.write(f' {post:.10f}')

    with open(output_file_grid,'a') as myfile:
        myfile.write(f'\n')

print("ALL DONE")
