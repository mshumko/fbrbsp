"""
Make a scatter plot between the chorus and microburst occurrence rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fbrbsp


fb_id = 3
rbsp_id = 'a'
dL = 1
dMLT = 1
microburst_version = 5
fb_time_window_sec = 2*60
dt = pd.Timedelta(seconds=fb_time_window_sec)

conjunction_name = (
    f'FU{fb_id}_RBSP{rbsp_id.upper()}_conjunctions_'
    f'dL{str(dL*10).zfill(2)}_dMLT{str(dMLT*10).zfill(2)}_final_hr.csv'
    )
conjunction_path = fbrbsp.config['here'].parent / 'data' / conjunction_name
conjunctions = pd.read_csv(conjunction_path)
for key in ['startTime', 'endTime']:
    conjunctions[key] = pd.to_datetime(conjunctions[key])

microburst_name = f'FU{fb_id}_microburst_catalog_{str(microburst_version).zfill(2)}.csv'
microburst_path = fbrbsp.config['here'].parent / 'data' / microburst_name
microbursts = pd.read_csv(microburst_path)
microbursts['Time'] = pd.to_datetime(microbursts['Time'])

conjunctions['microbursts'] = np.nan
conjunctions['microbursts_norm'] = np.nan

for i, conjunction in conjunctions.iterrows():
    n = np.where(
        (microbursts['Time'] > conjunction['startTime']-dt) & 
        (microbursts['Time'] <= conjunction['endTime']+dt)
        )[0].shape[0]
    conjunctions.loc[i, 'microbursts'] = n
    total_seconds = (conjunction['endTime']-conjunction['startTime']+2*dt).total_seconds()
    conjunctions.loc[i, 'microbursts_norm'] = n/total_seconds

# plt.hist(conjunctions.loc[:, 'microbursts'], bins=np.linspace(1, 50))
plt.hist(conjunctions.loc[:, 'microbursts_norm'], bins=np.linspace(0.001, 1))
plt.show()