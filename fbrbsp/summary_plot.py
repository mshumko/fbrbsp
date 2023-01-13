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


class Summary:
    def __init__(self, fb_id, fbsp_id, file_name, rbsp_xlabels=None) -> None:
        self.fb_id = fb_id 
        self.rbsp_id = fbsp_id
        self.file_name = file_name
        self.rbsp_xlabels = rbsp_xlabels
        if self.rbsp_xlabels is None:
            self.rbsp_xlabels = {"L": "L_90", "MLT": "MLT"}

        self.catalog_path = fbrbsp.config['here'].parent / 'data' / file_name
        self.catalog = pd.read_csv(self.catalog_path)
        self.catalog['startTime'] = pd.to_datetime(self.catalog['startTime'])
        self.catalog['endTime'] = pd.to_datetime(self.catalog['endTime'])

        self.save_path = fbrbsp.config['here'].parent / 'plots'
        if not self.save_path.exists():
            self.save_path.mkdir()
            print(f'Created plotting directory at {self.save_path}')
        pass

    def loop(self, survey_pad_min=60, zoom_pad_min=1):
        # self._init_plot()

        for i, (start_time, end_time) in enumerate(zip(self.catalog['startTime'], self.catalog['endTime'])):
            print(f'Processing conjunction: {start_time.isoformat()} ({i}/{self.catalog.shape[0]})')
            self._init_plot()
            survey_time_range = (
                start_time-timedelta(minutes=survey_pad_min/2),
                end_time+timedelta(minutes=survey_pad_min/2)
            )
            zoom_time_range = (
                start_time-timedelta(minutes=zoom_pad_min/2),
                end_time+timedelta(minutes=zoom_pad_min/2)
            )

            self._rbsp_magephem_labels(survey_time_range, self.ax[3,0])

            self._plot_magnetic_field(self.ax[0,0], self.ax[2,0], 
                survey_time_range, zoom_time_range
                )
            self._plot_electric_field(self.ax[1,0], self.ax[3,0], 
                survey_time_range, zoom_time_range
                )
            self._plot_firebird(self.ax[-1,0], zoom_time_range)

            self._plot_labels(zoom_time_range[0])

            save_name = (
                f'{start_time:%Y%m%d_%H%M%S}_{end_time:%H%M%S}_RBSP{self.rbsp_id.upper()}'
                f'_FB{self.fb_id}_conjunction_summary.png'
                )
            plt.savefig(self.save_path / save_name)
            # self._clear_plot()
            plt.close()
        return

    def _init_plot(self):
        self.n_rows = 6
        self.n_cols = 1
        self.fig = plt.figure(constrained_layout=False, figsize=(8, 10))
        spec = gridspec.GridSpec(nrows=self.n_rows, ncols=self.n_cols, figure=self.fig)
        self.ax = np.zeros((self.n_rows, self.n_cols), dtype=object)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.ax[i,j] = self.fig.add_subplot(spec[i, j])
        return
    
    def _plot_labels(self, date):
        plt.suptitle(
            f'{date:%F} RBSP{self.rbsp_id.upper()} - FU{self.fb_id}\n'
            f'conjunction summary'
            )
        self.ax[0,0].set_ylabel('Frequency')
        self.ax[1,0].set_ylabel('Frequency')
        self.ax[-1,0].set_ylabel('Collimated\n[counts]')
        self.ax[0,0].text(0, 0.99, 'EMFISIS WFR spectra', va='top', fontsize=15,
            c='g', transform=self.ax[0,0].transAxes)
        self.ax[1,0].text(0, 0.99, 'EFW spectra', va='top', fontsize=15,
            c='g', transform=self.ax[1,0].transAxes)
        self.ax[2,0].text(0, 0.99, 'EMFISIS WFR burst', va='top', fontsize=15,
            c='g', transform=self.ax[2,0].transAxes)
        self.ax[3,0].text(0, 0.99, 'EFW mscb1', va='top', fontsize=15,
            c='g', transform=self.ax[3,0].transAxes)
        self.ax[-1,0].text(0, 0.99, 'FIREBIRD', va='top', fontsize=15,
            c='g', transform=self.ax[-1,0].transAxes)
        return

    def _plot_magnetic_field(self, ax, bx, survey_time_range, zoom_time_range):

        emfisis_spec = Spec(self.rbsp_id, 'WFR', survey_time_range)
        emfisis_spec.load()
        plot_fce = True
        try:
            emfisis_spec.spectrum(ax=ax)
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

        emfisis_burst = Burst(self.rbsp_id, 'WFR', zoom_time_range)
        try:
            emfisis_burst.load()
        except (FileNotFoundError, ValueError) as err:
            if 'does not contain any hyper references' in str(err):
                return
            elif 'The EMFISIS burst data is incomplete.' in str(err):
                return
            else:
                raise
        try:
            emfisis_burst.spectrum(ax=bx, fce=plot_fce, 
                pcolormesh_kwargs={'norm':matplotlib.colors.LogNorm(vmin=1E-4, vmax=1E-3)})
        except ValueError as err:
            if 'No burst data' in str(err):
                return
            else:
                raise
        
        bx.set_ylim(
            np.min(emfisis_spec.WFR_frequencies), 
            np.max(emfisis_spec.WFR_frequencies)
        )
        bx.set_xlim(zoom_time_range)
        return

    def _plot_electric_field(self, ax, bx, survey_time_range, zoom_time_range):
        emfisis_spec = Spec(self.rbsp_id, 'WFR', survey_time_range)
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

    def _rbsp_magephem_labels(self, time_range, _ax):
        self.rbsp_magephem = MagEphem(self.rbsp_id, 't89d', time_range)
        self.rbsp_magephem.load()

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
                    self.rbsp_magephem[val[0]][i_min_time, ida].round(2).astype(str)
                    )
            else:
                tick_list.append(
                    self.rbsp_magephem[val][i_min_time].round(2).astype(str)
                    )
            
        # Cast np.array as strings so that it can insert the time string.
        tick_list = np.insert(tick_list, 0, 
            self.rbsp_magephem['epoch'][i_min_time].strftime("%H:%M:%S"))
        return "\n".join(tick_list)

    def _plot_firebird(self, ax, zoom_time_range):
        hr = Hires(self.fb_id, zoom_time_range[0]).load()
        for i in range(6):
            ax.plot(hr['Time'], hr['Col_counts'][:, i])
        ax.set_xlim(zoom_time_range)
        ax.set_yscale('log')
        return

    def _clear_plot(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.ax[i,j].clear()
        return
        
if __name__ == '__main__':
    fb_id = 3
    rbsp_id = 'A'
    for fb_id in [3, 4]:
        for rbsp_id in ['A', 'B']:
            file_name = f'FU{fb_id}_RBSP{rbsp_id.upper()}_conjunctions_dL10_dMLT10_final_hr.csv'

            s = Summary(fb_id, rbsp_id, file_name)
            # s.catalog = s.catalog[s.catalog.loc[:, 'startTime'] >= '2019-02-18']
            s.loop()