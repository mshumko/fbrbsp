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
fb_time_window_sec = 4*60
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
    microburst_dt = filtered_microburst_times.diff().dt.total_seconds()
    conjunctions.loc[i, 'mean_microburst_repetition'] = microburst_dt.mean()
    conjunctions.loc[i, 'median_microburst_repetition'] = microburst_dt.median()
    conjunctions.loc[i, 'std_microburst_repetition'] = microburst_dt.std()

fig, ax = plt.subplots()
ax.scatter(
    conjunctions['median_chorus_repetition'], 
    conjunctions['median_microburst_repetition'],
    c='k'
    )
plt.suptitle(f'FU-{fb_id} | RBSP-{rbsp_id.upper()}\nMedian chorus-microburst repetition rate')
ax.set_xlabel('Chorus repetition [sec]')
ax.set_ylabel('Microburst repetition [sec]')
max_repetition = np.max(np.concatenate((ax.get_xlim(), ax.get_ylim())))
ax.set_xlim(1E-1, max_repetition)
ax.set_ylim(1E-1, max_repetition)
ax.set_yscale('log')
ax.set_xscale('log')

ax.fill_between(
    np.linspace(0, max_repetition), 
    np.linspace(0, max_repetition), 
    max_repetition,
    color='blue', 
    alpha=0.2
    )
ax.text(
    0.02, 
    0.98, 
    'More chorus than microbursts', 
    color='k', 
    transform=ax.transAxes, 
    va='top', 
    fontsize=12
    )

ax.fill_between(
    np.linspace(0, max_repetition), 
    0, 
    np.linspace(0, max_repetition),
    color='orange', 
    alpha=0.2
    )
ax.text(
    0.98, 
    0.02, 
    'More microbursts than chorus', 
    color='k', 
    transform=ax.transAxes, 
    va='bottom', 
    ha='right', 
    fontsize=12
    )

plt.tight_layout()
plt.show()