"""
Plot a dispersed microburst given a time and a reference catalog.
"""
import dateutil.parser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates

import fbrbsp
import fbrbsp.load.firebird
import fbrbsp.duration.fit


time = '2015-08-27T12:41:01.663000'
plot_window_s=1
channels = np.arange(5)

# time = '2015-08-27T12:40:37'
# plot_window_s=2
# channels = np.arange(4)

# time = '2015-02-02T06:12:31.750000'
# plot_window_s=1

# time = '2015-02-02T06:12:26.310000'
# plot_window_s=1

fb_id = 3
catalog_version=5
fit_interval_s = 0.3


if isinstance(time, str):
    time = dateutil.parser.parse(time)
microburst_name = f'FU{fb_id}_microburst_catalog_{str(catalog_version).zfill(2)}.csv'
microburst_path = fbrbsp.config['here'].parent / 'data' / microburst_name
df = pd.read_csv(microburst_path)
df['Time'] = pd.to_datetime(df['Time'])
idt = np.argmin(np.abs(df['Time']-time))
microburst_info = df.loc[idt, :]

hr = fbrbsp.load.firebird.Hires(fb_id, time).load()
dt = pd.Timedelta(seconds=plot_window_s/2)
time_range = (microburst_info['Time']-dt, microburst_info['Time']+dt)
idt = np.where((hr['Time'] > time_range[0]) & (hr['Time'] < time_range[1]))[0]

_plot_colors = ['k', 'r', 'g', 'b', 'c', 'purple']
_, ax = plt.subplots(len(channels)+1, 1, figsize=(6, 8))
# ax[-1].get_shared_x_axes().remove(ax[-1])  # TODO: Find another solution to unshare x-axis of the last subplot.
for ax_i in ax[1:-1]:
    ax[0].get_shared_x_axes().join(ax[0], ax_i)
for ax_i in ax[:-2]:
    ax_i.xaxis.set_visible(False)


# Plot HiRes
for i, (color, channel) in enumerate(zip(_plot_colors, channels)):
    ax[i].step(hr['Time'][idt], hr['Col_counts'][idt, channel], c=color, where='post')
    energy_range = hr.attrs['Col_counts']['ENERGY_RANGES'][channel]
    ax[i].set_ylabel(f'{channel=}\n({energy_range})\nCounts/{1000*float(hr.attrs["CADENCE"])} ms')
    
# Plot Fit
time_array = hr['Time'][idt]
current_date = time_array[0].floor('d')
x_data_seconds = (time_array-current_date).total_seconds()
fit_interval_s = pd.Timedelta(seconds=fit_interval_s)
for i, (color, channel) in enumerate(zip(_plot_colors, channels)):
    fit_bounds = (
        microburst_info['Time']-fit_interval_s/2,
        microburst_info['Time']+fit_interval_s/2
    )
    ax[i].axvspan(*fit_bounds, color='grey', alpha=0.5)
    ax[i].axvline(microburst_info['Time'], color='k', ls=':')
    
    popt = np.nan*np.zeros(5)
    popt[1] = (dateutil.parser.parse(microburst_info[f't0_{channel}']) - current_date).total_seconds()
    popt[2] = microburst_info[f'fwhm_{channel}']/2.355 # Convert the Gaussian FWHM to std
    popt[0] = microburst_info[f'A_{channel}']
    popt[3] = microburst_info[f'y_int_{channel}']
    popt[4] = microburst_info[f'slope_{channel}']
    if np.isnan(popt[0]):  # Plot just the data if the fit failed.
        continue

    gaus_y = fbrbsp.duration.fit.Duration.gaus_lin_function(x_data_seconds, *popt)
    ax[i].plot(time_array, gaus_y, c=color, ls='--')

    max_counts = np.max(hr['Col_counts'][idt, channel])
    ax[i].set_ylim(0, 1.2*max_counts)

    fit_params=(
        f"FWHM={round(microburst_info[f'fwhm_{channel}'], 2)} [s]\n"
        f"R^2 = {round(microburst_info[f'r2_{channel}'], 2)}\n"
        f"adj_R^2 = {round(microburst_info[f'adj_r2_{channel}'], 2)}\n"
    )
    ax[i].text(0.01, 1, fit_params, va='top', transform=ax[i].transAxes, color=color)


locator=matplotlib.ticker.MaxNLocator(nbins=5)
ax[-2].xaxis.set_major_locator(locator)
fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
ax[-2].xaxis.set_major_formatter(fmt)

# Time differences with respect to channel 0
t0_keys = [f't0_{channel}' for channel in channels]
t0_0 = dateutil.parser.parse(microburst_info['t0_0'])
t0_differences_ms = [1E3*(t0_0 - dateutil.parser.parse(microburst_info[key])).total_seconds() for key in t0_keys]
center_energy = [float(s.split()[0].replace('>', '')) 
    for s in np.array(hr.attrs['Col_counts']['ELEMENT_LABELS'])[channels]]

xerrs = [xerr for xerr in np.array(hr.attrs['Col_counts']['ENERGY_RANGES'])[channels]]
xerrs = [xerr.replace('keV', '').replace('>', '').split('-') for xerr in xerrs]

if 5 in channels:  # Integral channel special case
    xerrs[-1] = [None, None]
xerrs = np.array(xerrs).astype(float).T - center_energy
xerrs = np.abs(xerrs)

yerrs = 1000*float(hr.attrs["CADENCE"])

s = (
        f'L={round(microburst_info["McIlwainL"], 1)}\n'
        f'MLT={round(microburst_info["MLT"], 1)}\n'
        f'(lat,lon)=({round(microburst_info["Lat"], 1)}, {round(microburst_info["Lon"], 1)})'
    )
ax[0].text(0.7, 1, s, va='top', transform=ax[0].transAxes, color='red')
ax[-1].errorbar(center_energy, t0_differences_ms, c='k', marker='.', 
    yerr=yerrs, xerr=xerrs, capsize=2, ls='None')
max_abs_lim = 1.1*np.max(np.abs(ax[-1].get_ylim()))
ax[0].set_title(f'Microburst dispersion\nFU{fb_id} | {microburst_info["Time"]}')
ax[-1].set_ylim(-max_abs_lim, max_abs_lim)
ax[-1].axhline(c='k', ls='--')
ax[-1].set(xlabel='Energy [keV]', ylabel='Peak time delay [ms]\n(ch[0]-ch[N])')

plt.tight_layout()
plt.show()