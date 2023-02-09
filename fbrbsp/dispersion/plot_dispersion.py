"""
Plot a dispersed microburst given a time and a reference catalog.
"""
import dateutil.parser
from datetime import datetime, date
from typing import Union
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

import fbrbsp
import fbrbsp.load.firebird
import fbrbsp.duration.fit


class Plot_Dispersion:
    def __init__(self, fb_id:int, channels:list=np.arange(6), 
                 catalog_version:int=5, fit_interval_s:float=0.3, 
                 plot_window_s:float=1):
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
        plot_window_s: float
            The plot window to use.

        Methods
        -------
        plot(time)
            Plot the HiRes data and peak dispersion.
        """
        self.fb_id = fb_id
        self.channels = channels
        self.catalog_version = catalog_version
        self.fit_interval_s = pd.Timedelta(seconds=fit_interval_s)
        self.plot_window_s = pd.Timedelta(seconds=plot_window_s)
        self.current_date = date.min
        self._plot_colors = np.array(['k', 'k', 'k', 'k', 'k', 'k'])

        catalog_name = f'FU{self.fb_id}_microburst_catalog_{str(self.catalog_version).zfill(2)}.csv'
        catalog_path = fbrbsp.config['here'].parent / 'data' / catalog_name
        self.catalog = pd.read_csv(catalog_path)
        self.catalog['Time'] = pd.to_datetime(self.catalog['Time'])
        return
    
    def plot(self, time:Union[str, datetime, pd.Timestamp], annotate_fit=True):
        """
        Plot the microburst dispersion.

        Parameters
        ----------
        time: str, datetime, or pd.Timestamp
            The time to reference and plot the microburst from the catalog. 
        annotate_fit: bool
            Annotate each subplot with the Gaussian FWHM and fit quality.
        """
        self._time = time
        self.annotate_fit = annotate_fit
        if isinstance(self._time, str):
            self._time = dateutil.parser.parse(self._time)
        idt = np.argmin(np.abs(self.catalog['Time']-self._time))
        self.microburst_info = self.catalog.loc[idt, :]
        dt = np.abs((self.microburst_info['Time']-self._time).total_seconds())
        if dt > 1:
            raise ValueError(f'The desired microburst plot time is {dt} '
                             f'seconds away from the closest microburst '
                             f'observed at {self.microburst_info["Time"]}')
        
        if self.current_date != self._time.date():
            self.hr = fbrbsp.load.firebird.Hires(self.fb_id, self._time).load()
            self.cadence_ms = 1000*float(self.hr.attrs["CADENCE"])
            self.center_energy, self.energy_range = self.get_energy_channels()
            self.current_date = self._time.date()

        time_range = (
            self.microburst_info['Time']-self.plot_window_s/2, 
            self.microburst_info['Time']+self.plot_window_s/2
            )
        self.plot_idt = np.where((self.hr['Time'] > time_range[0]) & (self.hr['Time'] < time_range[1]))[0]

        self._create_subplots()
        self._plot_hr()
        self._plot_fit()
        self._annotate_location()
        self._plot_dispersion(self.ax[-1])
        self.ax[0].set_title(f'FU{self.fb_id} Microburst Dispersion\n{self.microburst_info["Time"]}')
        self._format_times(self.ax[-2])
        self.ax[-2].set_xlabel('Time [HH:MM:SS]')
        return
    
    def _create_subplots(self):
        """"
        Create empty subplots for the HiRes line plots and dispersion scatter plot.
        """
        # I want to adjust the hspace for the HiRes line subplots and the dispersion 
        # subplot separately so I created multiple nested gridspecs.
        # See https://stackoverflow.com/a/31485288 for inspiration
        outer_gridspec = gridspec.GridSpec(2, 1, height_ratios=[len(self.channels), 1], 
                                           top=0.95, left=0.152, right=0.958, bottom=0.055, hspace=0.15) 
        inner_gs1 = gridspec.GridSpecFromSubplotSpec(len(self.channels), 1, subplot_spec=outer_gridspec[0], hspace=0.05)
        inner_gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gridspec[1])

        self.fig = plt.figure(figsize=(6, 8))
        self.ax = [None]*(len(self.channels)+1)
        for i in range(len(self.channels)):
            if i == 0:
                self.ax[i] = self.fig.add_subplot(inner_gs1[i, 0])
            else:
                self.ax[i] = self.fig.add_subplot(inner_gs1[i, 0], sharex=self.ax[0])
            if i < len(self.channels)-1:
                self.ax[i].get_xaxis().set_visible(False)
        self.ax[-1] = self.fig.add_subplot(inner_gs2[0, 0])
        for i, (ax_i, color) in enumerate(zip(self.ax, self._plot_colors[self.channels][::-1])):
            ax_i.text(0, 0.99, f'({string.ascii_uppercase[i]})', va='top', 
                      transform=ax_i.transAxes, weight='bold', color=color)
        return
    
    def _plot_hr(self):
        """
        Plot the Hires channels in each subplot.
        """
        # Plot HiRes
        for i, (ax_i, color, channel) in enumerate(zip(self.ax[:-1][::-1], self._plot_colors, self.channels)):
            ax_i.step(
                self.hr['Time'][self.plot_idt], 
                self.hr['Col_counts'][self.plot_idt, channel], c=color, where='mid'
                )
            _energy_range = self.energy_range[i].replace(' ', '')
            ax_i.set_ylabel(f'{channel=}\n({_energy_range})\n[counts/{self.cadence_ms} ms]')
            max_counts = np.max(self.hr['Col_counts'][self.plot_idt, channel])
            ax_i.set_ylim(0, 1.2*max_counts)
        return
    
    def _plot_fit(self):
        """
        Plot the Gaussian fit + linear trend
        """
        # Plot Fit
        data_time_array = self.hr['Time'][self.plot_idt]
        time_array = pd.date_range(start=data_time_array[0], end=data_time_array[-1], periods=1000)
        current_date = time_array[0].floor('d')
        x_data_seconds = (time_array-current_date).total_seconds()

        for i, (ax_i, color, channel) in enumerate(zip(self.ax[:-1][::-1], self._plot_colors, self.channels)):
            fit_bounds = (
                self.microburst_info['Time']-self.fit_interval_s/2,
                self.microburst_info['Time']+self.fit_interval_s/2
            )
            ax_i.axvspan(*fit_bounds, color='grey', alpha=0.5)
            ax_i.axvline(self.microburst_info['Time'], color='k', ls=':')
            
            popt = np.nan*np.zeros(5)
            popt[1] = (dateutil.parser.parse(self.microburst_info[f't0_{channel}']) - current_date).total_seconds()
            popt[2] = self.microburst_info[f'fwhm_{channel}']/2.355 # Convert the Gaussian FWHM to std
            popt[0] = self.microburst_info[f'A_{channel}']
            popt[3] = self.microburst_info[f'y_int_{channel}']
            popt[4] = self.microburst_info[f'slope_{channel}']
            if np.isnan(popt[0]):  # Plot just the data if the fit failed.
                continue

            gaus_y = fbrbsp.duration.fit.Duration.gaus_lin_function(x_data_seconds, *popt)
            ax_i.plot(time_array, gaus_y, c=color, ls='--')

            fit_params=(
                f"FWHM={round(self.microburst_info[f'fwhm_{channel}'], 2)} [s]\n"
                f"R^2 = {round(self.microburst_info[f'r2_{channel}'], 2)}\n"
                f"adj_R^2 = {round(self.microburst_info[f'adj_r2_{channel}'], 2)}\n"
            )
            ax_i.text(0.01, 0.87, fit_params, va='top', transform=ax_i.transAxes, color=color)
        return
    
    def _annotate_location(self):
        lat_str = f'${{{round(self.microburst_info["Lat"])}}}^{{\circ}}$'
        lon_str = f'${round(self.microburst_info["Lon"])}^{{\circ}}$'
        s = (
            f'L={round(self.microburst_info["McIlwainL"], 1)}\n'
            f'MLT={round(self.microburst_info["MLT"], 1)}\n'
            f'(lat,lon)=({lat_str},{lon_str})'
            )
        self.ax[0].text(0.67, 1, s, va='top', transform=self.ax[0].transAxes, color='k')
        return
    
    def _get_dispersion(self):
        # Time differences with respect to channel 0
        t0_keys = [f't0_{channel}' for channel in channels]
        t0 = [dateutil.parser.parse(self.microburst_info[t0_key]) for t0_key in t0_keys]
        self.t0_diff_ms = [1E3*(t0_i - t0[0]).total_seconds() for t0_i in t0]

        self.xerrs = [xerr for xerr in self.energy_range]
        self.xerrs = [xerr.replace('keV', '').replace('>', '').split('-') for xerr in self.xerrs]

        if 5 in self.channels:  # Integral channel special case
            self.xerrs[-1] = [None, None]
        self.xerrs = np.array(self.xerrs).astype(float).T - self.center_energy
        self.xerrs = np.abs(self.xerrs)
        self.yerrs = self.cadence_ms
        return
    
    def _plot_dispersion(self, ax):
        self._get_dispersion()
        ax.errorbar(self.center_energy, self.t0_diff_ms, c='k', marker='.', 
            yerr=self.yerrs, xerr=self.xerrs, capsize=2, ls='None')
        max_abs_lim = 1.1*np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-max_abs_lim, max_abs_lim)
        ax.axhline(c='k', ls='--')
        ax.set(xlabel='Energy [keV]', ylabel='Peak time delay [ms]\n(ch[N]-ch[0])')

        locator=matplotlib.ticker.FixedLocator(np.linspace(-max_abs_lim, max_abs_lim, num=5))
        ax.yaxis.set_major_locator(locator)
        return
    
    def get_energy_channels(self):
        center_energy = np.array([float(s.split()[0].replace('>', '')) 
                              for s in np.array(self.hr.attrs['Col_counts']['ELEMENT_LABELS'])])
        center_energy = center_energy[self.channels]
        energy_range = np.array(self.hr.attrs['Col_counts']['ENERGY_RANGES'])
        energy_range = energy_range[[self.channels]]
        return center_energy, energy_range
    
    def _format_times(self, ax):
        locator=matplotlib.ticker.MaxNLocator(nbins=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_fn))
        return

    def format_fn(self, tick_val, _):
        tick_time = matplotlib.dates.num2date(tick_val).replace(tzinfo=None)
        return tick_time.strftime('%T.%f')[:-5]


if __name__ == '__main__':
    plot_window_s=1

    # time = '2015-08-27T12:41:01.663000'
    # channels = np.arange(5)

    time = '2015-08-27T12:40:37'
    channels = np.arange(4)

    # time = '2015-02-02T06:12:31.750000'

    # time = '2015-02-02T06:12:26.310000'

    fb_id = 3
    catalog_version=5
    fit_interval_s = 0.3

    d = Plot_Dispersion(fb_id, channels=channels, catalog_version=catalog_version, 
                    fit_interval_s=fit_interval_s, plot_window_s=plot_window_s)
    d.plot(time)
    plt.tight_layout()
    plt.show()