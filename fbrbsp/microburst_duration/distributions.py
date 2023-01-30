"""
Plots the overall microburst duration distribution.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fbrbsp
import fbrbsp.load.firebird

fb_id = 3
catalog_version=5
max_width_ms = 500
r2_thresh = 0.9
width_energy_diff_percent = 20
width_bins = np.linspace(0, max_width_ms+0.001, num=25)

microburst_name = f'FU{fb_id}_microburst_catalog_{str(catalog_version).zfill(2)}.csv'
microburst_path = fbrbsp.config['here'].parent / 'data' / microburst_name
df = pd.read_csv(microburst_path)
# Drop microbursts that were not fit for one reason or another.
df = df.dropna().reset_index()

# Filter out microbursts where none of the energy channels had an r2>r2_thresh
# and the fwhm were not within a % of each other.
r2_keys = [key for key in df.columns if 'r2' == key.split('_')[0]]
fwhm_keys = [key for key in df.columns if 'fwhm' == key.split('_')[0]]
n_nan_start = sum(np.isnan(df[fwhm_keys[0]]))
df['max_r2'] = df[r2_keys].max(axis=1)
df.loc[df['max_r2'] < r2_thresh, :] = np.nan
df = df.dropna().reset_index()
# for i, row in df.iterrows():
#     fwhm_dispersion_percent = np.abs(
#         row[fwhm_keys].mean() - row[fwhm_keys].std()
#         )/row[fwhm_keys].mean()*100
#     # if ((fwhm_dispersion_percent > width_energy_diff_percent) or 
#     #     (row[r2_keys].max() < r2_thresh)):
#     #     df.loc[i, :] = np.nan
#     if row[r2_keys].max() < r2_thresh:
#         df.loc[i, :] = np.nan
print(f"Number of fits NaN'd {n_nan_start - sum(np.isnan(df[fwhm_keys[0]]))}")

    

# For the energy channel information
hr = fbrbsp.load.firebird.Hires(fb_id, '2015-02-02').load()
fig, ax = plt.subplots(6,1, sharex=True, figsize=(6, 8.8))

for ch in range(6):
    width_key = f'fwhm_{ch}'
    df_flt = df.copy()
    df_flt[width_key] = 1000*df_flt[width_key]
    # df_flt = df_flt[df_flt[width_key] < max_width_ms]
    # df_flt = df_flt[df_flt[f'adj_r2_{ch}'] > r2_thresh]

    print(f'Channel {ch}, {sum(~np.isnan(df[width_key]))} total microbursts fit')
    print(f'{df_flt.shape[0]} well-fit microbursts.')

    quantiles = [.25, .50, .75]
    width_percentiles = df_flt[width_key].quantile(q=quantiles)

    ax[ch].hist(df_flt[width_key], bins=width_bins, color='k', histtype='step', density=True)
    s = (
        f"Channel {ch} ({hr.attrs['Col_counts']['ENERGY_RANGES'][ch]})\n"
        f"Percentiles [ms]"
        f"\n25%: {(width_percentiles.loc[0.25]).round().astype(int)}"
        f"\n50%: {(width_percentiles.loc[0.50]).round().astype(int)}"
        f"\n75%: {(width_percentiles.loc[0.75]).round().astype(int)}"
    )
    ax[ch].text(0.50, 0.9, s, 
            ha='left', va='top', transform=ax[ch].transAxes
            )
    plt.suptitle(f'Distribution of Microburst Duration | FU{fb_id}')
    ax[ch].set_xlim(0, max_width_ms)

ax[2].set_ylabel('Probability Density')
ax[-1].set_xlabel('FWHM [ms]')
plt.tight_layout()
plt.show()