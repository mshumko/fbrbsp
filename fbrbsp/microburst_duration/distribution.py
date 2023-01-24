"""
Plots the overall microburst duration distribution.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fbrbsp

fb_id = 3
catalog_version=5
max_width_ms = 500
r2_thresh = 0.8
ch = 0
width_key = f'fwhm_{ch}'
width_bins = np.linspace(0, max_width_ms+0.001, num=25)

microburst_name = f'FU{fb_id}_microburst_catalog_{str(catalog_version).zfill(2)}.csv'
microburst_path = fbrbsp.config['here'].parent / 'data' / microburst_name
df = pd.read_csv(microburst_path)
print(f'{sum(~np.isnan(df[width_key]))} total microbursts fit')

df[width_key] = 1000*df[width_key]
df = df[df[width_key] < max_width_ms]
df = df[df[f'r2_{ch}'] > r2_thresh]

print(f'{df[width_key].shape[0]} well-fit microbursts.')

quantiles = [.25, .50, .75]
width_percentiles = df[width_key].quantile(q=quantiles)

fig, ax = plt.subplots()
ax.hist(df[width_key], bins=width_bins, color='k', histtype='step', density=True)
s = (
    f"Percentiles [ms]"
    f"\n25%: {(width_percentiles.loc[0.25]).round().astype(int)}"
    f"\n50%: {(width_percentiles.loc[0.50]).round().astype(int)}"
    f"\n75%: {(width_percentiles.loc[0.75]).round().astype(int)}"
)
ax.text(0.64, 0.9, s, 
        ha='left', va='top', transform=ax.transAxes
        )
plt.suptitle(f'Distribution of ~200 keV Microburst Duration | FU{fb_id}')
# Left panel tweaks
ax.set_xlim(0, max_width_ms)
ax.set_ylabel('Probability Density')
ax.set_xlabel('FWHM [ms]')
plt.show()