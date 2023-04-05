"""
Make a scatter plot between the chorus and microburst occurrence rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fbrbsp
import fbrbsp.load.firebird


fb_id = 3
rbsp_id = 'a'
dL = 1
dMLT = 1
microburst_version = 5
fb_time_window_sec = 2*60
dt = pd.Timedelta(seconds=fb_time_window_sec)

# conjunction_name = (
#     f'FU{fb_id}_RBSP{rbsp_id.upper()}_conjunctions_'
#     f'dL{str(dL*10).zfill(2)}_dMLT{str(dMLT*10).zfill(2)}_final_hr.csv'
#     )
conjunction_name = f'c_rp_FU{fb_id}_RBSP{rbsp_id.upper()}.csv'
conjunction_path = fbrbsp.config['here'].parent / 'data' / conjunction_name
conjunctions = pd.read_csv(conjunction_path)
for key in ['start_time', 'end_time']:
    conjunctions[key] = pd.to_datetime(conjunctions[key])

microburst_name = f'FU{fb_id}_microburst_catalog_{str(microburst_version).zfill(2)}.csv'
microburst_path = fbrbsp.config['here'].parent / 'data' / microburst_name
microbursts = pd.read_csv(microburst_path)
microbursts['Time'] = pd.to_datetime(microbursts['Time'])

conjunctions['n_microbursts'] = np.nan
conjunctions['mean_microburst_repetition'] = np.nan
conjunctions['microbursts_std'] = np.nan

for i, conjunction in conjunctions.iterrows():
    print(f'Processing conjunctions {round(100*i/conjunctions.shape[0])}%.')
    microburst_idx =  np.where(
        (microbursts['Time'] > conjunction['start_time']-dt) & 
        (microbursts['Time'] <= conjunction['end_time']+dt)
        )[0]
    conjunctions.loc[i, 'n_microbursts'] = microburst_idx.shape[0]

    # Load FIREBIRD data to calculate how long FIREBIRD took hr data in 
    # between (conjunction['start_time']-dt, conjunction['start_time']+dt).
    hr = fbrbsp.load.firebird.Hires(fb_id, conjunction['start_time']).load()
    hr_idx =  np.where(
        (hr['Time'] > conjunction['start_time']-dt) & 
        (hr['Time'] <= conjunction['end_time']+dt)
        )[0]
    total_seconds = (hr['Time'][hr_idx[-1]]-hr['Time'][hr_idx[0]]).total_seconds()
    conjunctions.loc[i, 'mean_microburst_repetition'] = conjunctions.loc[i, 'n_microbursts']/total_seconds

    filtered_microburst_times = microbursts.loc[microburst_idx, 'Time']
    if conjunctions.loc[i, 'n_microbursts']:
        microburst_dt = [(t_next-t_current).total_seconds() for t_current, t_next in 
                         zip(filtered_microburst_times.iloc[:-1], filtered_microburst_times.iloc[1:])]
        conjunctions.loc[i, 'microbursts_std'] = np.std(microburst_dt)

fig, ax = plt.subplots(2, figsize=(8, 5))
# ax.axis('equal')
ax[0].scatter(conjunctions['n_chorus'], conjunctions['n_microbursts'])
ax[0].set_xlabel('n_chorus')
ax[0].set_ylabel('n_microbursts')
ax[1].scatter(conjunctions['mean_chorus_repetition'], conjunctions['mean_microburst_repetition'])
ax[1].set_xlabel('mean_chorus_repetition')
ax[1].set_ylabel('mean_microburst_repetition')
plt.tight_layout()
plt.show()
pass