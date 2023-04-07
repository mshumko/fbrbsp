"""
Make a scatter plot between the chorus and microburst occurrence rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fbrbsp


debug = False
fb_id = 3
rbsp_id = 'a'
dL = 1
dMLT = 1
microburst_version = 5
fb_time_window_sec = 2*60
dt = pd.Timedelta(seconds=fb_time_window_sec)

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
conjunctions['median_microburst_repetition'] = np.nan
conjunctions['std_microburst_repetition'] = np.nan

for i, conjunction in conjunctions.iterrows():
    print(f'Processing conjunctions {round(100*i/conjunctions.shape[0])}%.')
    microburst_idx =  np.where(
        (microbursts['Time'] > conjunction['start_time']-dt) & 
        (microbursts['Time'] <= conjunction['end_time']+dt)
        )[0]
    conjunctions.loc[i, 'n_microbursts'] = microburst_idx.shape[0]

    if microburst_idx.shape[0] < 2:
        continue

    filtered_microburst_times = microbursts.loc[microburst_idx, 'Time']
    dt = filtered_microburst_times.diff().dt.total_seconds()
    conjunctions.loc[i, 'mean_microburst_repetition'] = dt.mean()
    conjunctions.loc[i, 'median_microburst_repetition'] = dt.median()
    conjunctions.loc[i, 'std_microburst_repetition'] = dt.std()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(conjunctions['n_chorus'], conjunctions['n_microbursts'])
ax[0].plot(np.linspace(0, ax[0].get_xlim()[1]), np.linspace(0, ax[0].get_xlim()[1]))
ax[0].set_xlabel('n_chorus')
ax[0].set_ylabel('n_microbursts')

ax[1].scatter(conjunctions['median_chorus_repetition'], conjunctions['median_microburst_repetition'])
ax[1].plot(np.linspace(0, ax[1].get_xlim()[1]), np.linspace(0, ax[1].get_xlim()[1]))
ax[1].set_xlabel('median_chorus_repetition')
ax[1].set_ylabel('median_microburst_repetition')

plt.suptitle(f'Chorus-Microburst repetition rate | FU-{fb_id} | RBSP-{rbsp_id.upper()}')
plt.tight_layout()

for ax_i in ax:
    ax_i.set_xlim(0, None)
    ax_i.set_ylim(0, None)
    # ax_i.axis('equal')

plt.show()
pass