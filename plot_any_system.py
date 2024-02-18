import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path = os.path.join(sys.argv[1],'')




r_le,p_le = np.loadtxt(path+"le_dps_ratios.dat",unpack=True)
r_in,p_in = np.loadtxt(path+"in_dps_ratios.dat",unpack=True)
r_co,p_co = np.loadtxt(path+"co_dps_ratios.dat",unpack=True)



fig,ax = plt.subplots(figsize=(10,6.18),nrows=1,ncols=1,gridspec_kw={'hspace':0.0,'wspace':0.0},sharey=True,sharex=True)

ax.plot(r_le,p_le,label="Lensing",color="red")
ax.plot(r_in,p_in,label="Intrinsic",color="blue")
ax.plot(r_co,p_co,label="Combined",color="black")

ax.axvline(1.0,color='gray',linestyle='-')

# cosmetics
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.02))
ax.set_xlim(0.0,2.0)
ax.set_ylim(bottom=0)
ax.set_ylim(top=6)
ax.set(xlabel=r'$W_0/W$',ylabel='dP')
ax.legend(loc='upper right')

fig.savefig("le_in_co_prob.pdf")
