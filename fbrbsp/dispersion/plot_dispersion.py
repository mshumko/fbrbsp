"""
Plot a dispersed microburst given a time and a reference catalog.
"""
import dateutil.parser
from datetime import datetime, date
from typing import Union

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


class Plot_Dispersion:
    def __init__(self, fb_id:int, channels:list=np.arange(6), catalog_version:int=5, fit_interval_s:float=0.3):
        """
        Plot the microburst HiRes data and dispersion.

        Parameters
        ----------
        fb_id: int
            The FIREBIRD spacecraft id. Can be 3 or 4
        channels: list or np.array
            The energy channels to plot.
        catalog_version: int 
            The microburst catalog version located in fbrbsp/data/
        fit_interval_s: float
            The fit interval used to fit microbursts in fit.py

        Methods
        -------
        plot(time)
            Plot the HiRes data and peak dispersion.
        """
        self.fb_id = fb_id
        self.channels = channels
        self.catalog_version = catalog_version
        self.fit_interval_s = fit_interval_s
        self.current_date = date.min

        catalog_name = f'FU{self.fb_id}_microburst_catalog_{str(self.catalog_version).zfill(2)}.csv'
        catalog_path = fbrbsp.config['here'].parent / 'data' / catalog_name
        self.catalog = pd.read_csv(catalog_path)
        self.catalog['Time'] = pd.to_datetime(self.catalog['Time'])
        return
    
    def plot(self, time:Union[str, datetime, pd.Timestamp]):
        """
        Plot the microburst dispersion.

        Parameters
        ----------
        time: str, datetime, or pd.Timestamp
            The time to reference and plot the microburst from the catalog. 
        """
        self._time = time
        if isinstance(self._time, str):
            self._time = dateutil.parser.parse(self._time)
        idt = np.argmin(np.abs(self.catalog['Time']-time))
        microburst_info = self.catalog.loc[idt, :]
        dt = np.abs((microburst_info['Time']-time).total_seconds())
        if dt > 1:
            raise ValueError(f'The desired microburst plot time is {dt} '
                             f'seconds away from the closest microburst '
                             f'observed at {microburst_info["Time"]}')
        
        if self.current_date != self._time.date():
            self.hr = fbrbsp.load.firebird.Hires(self.fb_id, self._time).load()
            self.current_date = self._time.date()

        plot_dt = pd.Timedelta(seconds=plot_window_s/2)
        time_range = (microburst_info['Time']-plot_dt, microburst_info['Time']+plot_dt)
        idt = np.where((self.hr['Time'] > time_range[0]) & (self.hr['Time'] < time_range[1]))[0]

        _plot_colors = ['k', 'r', 'g', 'b', 'c', 'purple']
        self.fig, self.ax = plt.subplots(len(self.channels)+1, 1, figsize=(6, 8))
        # ax[-1].get_shared_x_axes().remove(ax[-1])  # TODO: Find another solution to unshare x-axis of the last subplot.
        for ax_i in self.ax[1:-1]:
            self.ax[0].get_shared_x_axes().join(self.ax[0], ax_i)
        for ax_i in self.ax[:-2]:
            ax_i.xaxis.set_visible(False)
        return
    
    def _plot_hr(self):


        return
    

if __name__ == '__main__':


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
fmt = matplotlib.dates.DateFormatter('%H:%M:%S')  # Replace time ticks ms?
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

# for ax_i in ax[:-1]:
#     ax_i.set_ylim(0.1, None)
#     ax_i.set_yscale('log')

# print(f'{np.array(hr.attrs["Col_counts"]["ENERGY_RANGES"])[channels]=}')
# print(f'{t0_differences_ms=}')

plt.tight_layout()
plt.show()