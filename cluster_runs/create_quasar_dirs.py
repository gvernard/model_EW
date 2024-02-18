import os
import sys
import json
import numpy as np
import pandas as pd
from astropy.time import Time

path_to_input = sys.argv[1]
path_to_dirs = os.path.join(sys.argv[2],'')

# Read as a simple list
#sdss_data = np.loadtxt(path_to_input,delimiter=',',skiprows=1,usecols=[0,1,2,3,4,6],max_rows=10)
#nrows,ncols = sdss_data.shape

# Read as a pandas data frame, perform selections, and cast to list
df = pd.read_csv(path_to_input)
#sel_df = df.loc[df['EW'] < 6].head(10)
sel_df = df.sort_values(by=['EW']).head(2400)
sdss_data = sel_df.to_numpy()
nrows,ncols = sdss_data.shape




# Get future time
t = '2021-01-10T00:00:00'
time = Time(t,format='isot',scale='utc')
time_mjd = time.mjd
#print(time_mjd)

imf_type = "chabrier"
lrest = 5007

# Loop over systems and create directories and json files
for i in range(0,nrows):
    out_json = {}
    out_json['ra'] = sdss_data[i][0]
    out_json['dec'] = sdss_data[i][1]
    out_json['t'] = (time_mjd - sdss_data[i][2])/365.0
    out_json['zs'] = sdss_data[i][3]
    out_json['Mi'] = sdss_data[i][4]
    out_json['W0'] = sdss_data[i][5]
    out_json['imf_type'] = imf_type
    out_json['lrest'] = lrest

    
    qso_path = path_to_dirs+str(i).zfill(5)
    if not os.path.exists(qso_path):
        os.mkdir(qso_path)
    with open(qso_path+'/input_sdss.json','w') as fp:
        json.dump(out_json,fp)

