import pathlib
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

import fbrbsp
from fbrbsp.load.firebird import Hires
from fbrbsp.load.emfisis import Spec, Burst
from fbrbsp.load.rbsp_magephem import MagEphem
from fbrbsp.dial import Dial


class Summary:
    def __init__(self, fb_id, fbsp_id, file_name, rbsp_xlabels=None, fb_xlabels=None) -> None:
        self.fb_id = fb_id 
        self.rbsp_id = fbsp_id
        self.file_name = file_name
        self.rbsp_xlabels = rbsp_xlabels
        self.fb_xlabels = fb_xlabels
        if self.rbsp_xlabels is None:
            self.rbsp_xlabels = {"L": "L_90", "MLT": "EDMAG_MLT", f'$\\lambda$ [deg]':'EDMAG_MLAT'}
        if self.fb_xlabels is None:
            self.fb_xlabels = {"L": "McIlwainL", "MLT": "MLT", 'Lat [deg]': 'Lat', 'Lon [deg]':'Lon'}

        self.catalog_path = fbrbsp.config['here'].parent / 'data' / file_name
        self.catalog = pd.read_csv(self.catalog_path)
        self.catalog['startTime'] = pd.to_datetime(self.catalog['startTime'])
        self.catalog['endTime'] = pd.to_datetime(self.catalog['endTime'])

        self.save_path = fbrbsp.config['here'].parent / 'plots'
        if not self.save_path.exists():
            self.save_path.mkdir()
            print(f'Created plotting directory at {self.save_path}')
        pass

    def loop(self, zoom_pad_min=5, inspect=False):
        """
        The main method to create summary plots.
        """
        for i, (start_time, end_time) in enumerate(zip(self.catalog['startTime'], self.catalog['endTime'])):
            print(f'Processing RBSP{self.rbsp_id}-FU{self.fb_id} conjunction: '
                  f'{start_time.isoformat()} ({i}/{self.catalog.shape[0]})')
            self._init_plot()
            time_range = (
                start_time-timedelta(minutes=zoom_pad_min/2),
                end_time+timedelta(minutes=zoom_pad_min/2)
            )
            self.rbsp_magephem = MagEphem(self.rbsp_id, 't89d', time_range)
            self.rbsp_magephem.load()

            self._plot_magnetic_field(self.ax[0], time_range)
            self._plot_electric_field(self.ax[1], time_range)
            self._plot_firebird(self.ax[-1], time_range)
            self._plot_orbit(self.bx, time_range)
            self._plot_L_intersection(self.ax[-1], time_range)

            self._plot_labels(time_range[0])
            self._rbsp_magephem_labels(self.ax[1], time_range)
            self._fb_magephem_labels(self.ax[-1], time_range)

            if inspect:
                plt.show()
                continue

            save_name = (
                f'{start_time:%Y%m%d_%H%M%S}_{end_time:%H%M%S}_rbsp{self.rbsp_id.lower()}'
                f'_fb{self.fb_id}_conjunction_summary.png'
                )
            plt.subplots_adjust(hspace=0.795)
            plt.savefig(self.save_path / save_name)
            plt.close()
        return

    def _init_plot(self, n_rbsp_subplots=2, n_fb_subplots=1):
        if n_fb_subplots != 1:
            raise NotImplementedError
        # I want to adjust the hspace for the HiRes line subplots and the dispersion 
        # subplot separately so I created multiple nested gridspecs.
        # See https://stackoverflow.com/a/31485288 for inspiration
        n_total_subplots = n_rbsp_subplots + n_fb_subplots
        outer_gridspec = gridspec.GridSpec(2, 4, 
            height_ratios=[n_rbsp_subplots/n_total_subplots, n_fb_subplots/n_total_subplots], 
            top=0.94, left=0.1, right=0.958, bottom=0.13, hspace=0.27) 
        inner_gs1 = gridspec.GridSpecFromSubplotSpec(n_rbsp_subplots, 1, subplot_spec=outer_gridspec[0, :-1], hspace=0.05)
        inner_gs2 = gridspec.GridSpecFromSubplotSpec(n_fb_subplots, 1, subplot_spec=outer_gridspec[1, :-1])

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = [None]*n_total_subplots
        for i in range(n_rbsp_subplots):
            if i == 0:
                self.ax[i] = self.fig.add_subplot(inner_gs1[i, 0])
            else:
                self.ax[i] = self.fig.add_subplot(inner_gs1[i, 0], sharex=self.ax[0])
        self.ax[-1] = self.fig.add_subplot(inner_gs2[0])
        self.bx = self.fig.add_subplot(outer_gridspec[:, -1], projection='polar')  # Polar L-MLT plot.
        return
    
    def _plot_labels(self, date):
        plt.suptitle(f'{date:%F} RBSP{self.rbsp_id.upper()} - FU{self.fb_id} conjunction')
        self.ax[0].set_ylabel('Frequency')
        self.ax[0].set_yscale('log')
        self.ax[1].set_ylabel('Frequency')
        self.ax[1].set_yscale('log')
        self.ax[-1].set_ylabel('Collimated\n[counts]')

        self.ax[0].text(0, 0.99, 'EMFISIS WFR B spectra', va='top', fontsize=15,
            c='w', transform=self.ax[0].transAxes)
        self.ax[1].text(0, 0.99, 'EMFISIS WFR E spectra', va='top', fontsize=15,
            c='w', transform=self.ax[1].transAxes)
        self.ax[-1].text(0, 0.99, f'FIREBIRD Flight Unit {self.fb_id}', va='top', fontsize=15,
            c='k', transform=self.ax[-1].transAxes)
        return

    def _plot_magnetic_field(self, ax, time_range):
        """
        Plot the magnetic field spectrum.
        """
        emfisis_spec = Spec(self.rbsp_id, 'WFR', time_range)
        emfisis_spec.load()
        plot_fce = True
        try:
            emfisis_spec.spectrum(ax=ax)
            ax.legend(loc='lower right', fontsize='small', facecolor='grey', labelcolor='w')
        except ValueError as err: 
            if "Variable name 'Magnitude' not found." in str(err):
                plot_fce = False
                emfisis_spec.spectrum(ax=ax, fce=plot_fce)
            else:
                raise
        ax.set_ylim(
            np.min(emfisis_spec.WFR_frequencies), 
            np.max(emfisis_spec.WFR_frequencies)
        )
        return

    def _plot_electric_field(self, ax, time_range):
        emfisis_spec = Spec(self.rbsp_id, 'WFR', time_range)
        emfisis_spec.load()
        plot_fce = True
        try:
            emfisis_spec.spectrum(ax=ax, component='EuEu')
        except ValueError as err: 
            if "Variable name 'Magnitude' not found." in str(err):
                plot_fce = False
                emfisis_spec.spectrum(ax=ax, fce=plot_fce)
            else:
                raise
        ax.set_ylim(
            np.min(emfisis_spec.WFR_frequencies), 
            np.max(emfisis_spec.WFR_frequencies)
        )
        return

    def _plot_orbit(self, ax, time_range, L_labels=[2,4,6,8], max_L=10):
        """
        Make an orbit dial plot.
        """
        self._dial = Dial(ax, None, None, None)
        self._dial.L_labels = L_labels
        # Turn off the grid to prevent a matplotlib deprecation warning 
        # (see https://matplotlib.org/3.5.1/api/prev_api_changes/api_changes_3.5.0.html#auto-removal-of-grids-by-pcolor-and-pcolormesh)
        ax.grid(False) 
        self._dial.draw_earth()
        self._dial._plot_params()

        if not hasattr(self, 'hr'):
            raise ValueError('Need to call the _plot_firebird() method first.')
        
        fb_L, fb_mlt = self._fb_filtered_magephem(time_range)
        rb_L, rb_mlt = self._rbsp_filtered_magephem(time_range)
        
        ax.plot((2*np.pi/24)*fb_mlt, fb_L, 'k')
        ax.plot((2*np.pi/24)*rb_mlt, rb_L, marker='X', color='r')
        return
    
    def _plot_L_intersection(self, ax, time_range):
        """
        Plot a vertical black line in the FIREBIRD data panel at the time when
        the FIREBIRD's L-shell is closest to RBSP's L-shell.
        """
        rb_L, _ = self._rbsp_filtered_magephem(time_range)
        median_rb_L = np.nanmedian(rb_L)

        hr_idx = np.where(
            (self.hr['Time']>time_range[0]) & 
            (self.hr['Time']<=time_range[1])
            )[0]
        _hr_time = self.hr['Time'][hr_idx]
        _hr_L = self.hr['McIlwainL'][hr_idx]

        idx = np.nanargmin(np.abs(median_rb_L-_hr_L))
        ax.axvline(_hr_time[idx], c='k', ls=':')
        return
    
    def _rbsp_magephem_labels(self, _ax, time_range):
        _ax.xaxis.set_major_formatter(FuncFormatter(self.format_rbsp_xaxis))
        _ax.set_xlabel("\n".join(["Time"] + list(self.rbsp_xlabels.keys())))
        _ax.xaxis.set_label_coords(-0.1, -0.06)

        _ax.format_coord = lambda x, y: "{}, {}".format(
            matplotlib.dates.num2date(x).replace(tzinfo=None).isoformat(), round(y)
        )
        return

    def format_rbsp_xaxis(self, tick_val, tick_pos):
        """
        The tick magic happens here. pyplot gives it a tick time, and this function 
        returns the closest label to that time. Read docs for FuncFormatter().
        """
        # Find the nearest time within 30 seconds (the cadence of the RBSP mag ephem is 1 minute)
        tick_time = matplotlib.dates.num2date(tick_val).replace(tzinfo=None)
        i_min_time = np.argmin(np.abs(self.rbsp_magephem['epoch'] - tick_time))
        if np.abs(self.rbsp_magephem['epoch'][i_min_time] - tick_time).total_seconds() > 30:
            return tick_time.strftime("%H:%M:%S")

        # Construct the tick
        tick_list = []
        for key, val in self.rbsp_xlabels.items():
            if key[0].upper() == 'L':
                # find pitch angle
                ida = np.where(val.upper() == self.rbsp_magephem['L_Label'])[0]
                tick_list.append(
                    self.rbsp_magephem[val[0]][i_min_time, ida][0].round(2).astype(str)
                    )
            else:
                tick_list.append(
                    self.rbsp_magephem[val][i_min_time].round(2).astype(str)
                    )
            
        tick_list.insert(0, self.rbsp_magephem['epoch'][i_min_time].strftime("%H:%M:%S"))
        return "\n".join(tick_list)

    def _fb_magephem_labels(self, _ax, time_range):
        _ax.xaxis.set_major_formatter(FuncFormatter(self.format_fb_xaxis))
        _ax.set_xlabel("\n".join(["Time"] + list(self.fb_xlabels.keys())))
        _ax.xaxis.set_label_coords(-0.1, -0.06)

        _ax.format_coord = lambda x, y: "{}, {}".format(
            matplotlib.dates.num2date(x).replace(tzinfo=None).isoformat(), round(y)
        )
        return

    def format_fb_xaxis(self, tick_val, tick_pos):
        """
        The tick magic happens here. pyplot gives it a tick time, and this function 
        returns the closest label to that time. Read docs for FuncFormatter().
        """
        # Find the nearest time within 1 second.
        tick_time = matplotlib.dates.num2date(tick_val).replace(tzinfo=None)
        i_min_time = np.argmin(np.abs(self.hr['Time'] - tick_time))
        if np.abs(self.hr['Time'][i_min_time] - tick_time).total_seconds() > 1:
            return tick_time.strftime("%H:%M:%S")

        # Construct the tick
        tick_list = []
        for val in self.fb_xlabels.values():
            tick_list.append(self.hr[val][i_min_time].round(2).astype(str))
        tick_list.insert(0, self.hr['Time'][i_min_time].strftime("%H:%M:%S"))
        return "\n".join(tick_list)

    def _plot_firebird(self, ax, time_range):
        self.hr = Hires(self.fb_id, time_range[0]).load()
        for i in range(6):
            ax.plot(self.hr['Time'], self.hr['Col_counts'][:, i], 
                    label=self.hr.attrs['Col_counts']['ENERGY_RANGES'][i])
        ax.set_xlim(time_range)
        ax.set_yscale('log')
        hr_idx = np.where(
            (self.hr['Time']>time_range[0]) & 
            (self.hr['Time']<=time_range[1])
            )[0]
        try:
            time_correction = round(self.hr["Count_Time_Correction"][hr_idx].mean())
        except ValueError as err:
            if 'cannot convert float NaN to integer' in str(err):
                time_correction = 'NaN'
            elif 'All-NaN slice encountered' in str(err):
                time_correction = 'NaN'
            else:
                raise
        ax.text(0.99, 0.99, f'Time correction={time_correction} s', va='top', ha='right', 
            c='k', transform=ax.transAxes)
        ax.legend(loc='lower right', fontsize='small')
        return

    def _clear_plot(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.ax[i,j].clear()
        return

    def _fb_filtered_magephem(self, time_range, max_L=10):
        hr_idx = np.where(
            (self.hr['Time']>time_range[0]) & 
            (self.hr['Time']<=time_range[1]) &
            (self.hr['McIlwainL'] < max_L)
            )[0]
        fb_mlt = self.hr['MLT'][hr_idx]
        fb_L = self.hr['McIlwainL'][hr_idx]
        return fb_L, fb_mlt

    def _rbsp_filtered_magephem(self, time_range):
        rb_idx = np.where(
            (self.rbsp_magephem['epoch']>time_range[0]) & 
            (self.rbsp_magephem['epoch']<=time_range[1])
            )[0]
        ida = np.where(self.rbsp_xlabels['L'].upper() == self.rbsp_magephem['L_Label'])[0]
        rb_mlt = self.rbsp_magephem[self.rbsp_xlabels['MLT']][rb_idx]
        rb_L = self.rbsp_magephem['L'][rb_idx, ida]
        return rb_L, rb_mlt

if __name__ == '__main__':
    for fb_id in [3, 4]:
        for rbsp_id in ['A', 'B']:
            file_name = f'FU{fb_id}_RBSP{rbsp_id.upper()}_conjunctions_dL10_dMLT10_final_hr.csv'

            s = Summary(fb_id, rbsp_id, file_name)
            # s.catalog = s.catalog[s.catalog.loc[:, 'startTime'] >= '2019-02-18']
            s.loop()