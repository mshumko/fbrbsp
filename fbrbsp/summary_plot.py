import pathlib
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import fbrbsp
from fbrbsp.load.firebird import Hires
from fbrbsp.load.emfisis import Spec


class Summary:
    def __init__(self, fb_id, fbsp_id, file_name) -> None:
        self.fb_id = fb_id 
        self.rbsp_id = fbsp_id
        self.file_name = file_name

        self.catalog_path = fbrbsp.config['here'].parent / 'data' / file_name
        self.catalog = pd.read_csv(self.catalog_path)
        self.catalog['startTime'] = pd.to_datetime(self.catalog['startTime'])
        self.catalog['endTime'] = pd.to_datetime(self.catalog['endTime'])

        self.save_path = fbrbsp.config['here'].parent / 'plots'
        if not self.save_path.exists():
            self.save_path.mkdir()
            print(f'Created plotting directory at {self.save_path}')
        pass

    def loop(self, survey_pad_min=60, zoom_pad_min=6):
        self._init_plot()

        for start_time, end_time in zip(self.catalog['startTime'], self.catalog['endTime']):
            survey_time_range = (
                start_time-timedelta(minutes=survey_pad_min/2),
                end_time+timedelta(minutes=survey_pad_min/2)
            )
            zoom_time_range = (
                start_time-timedelta(minutes=zoom_pad_min/2),
                end_time+timedelta(minutes=zoom_pad_min/2)
            )
            emfisis_spec = Spec(self.rbsp_id, 'WFR', survey_time_range)
            emfisis_spec.load()
            emfisis_spec.spectrum(ax=self.ax[0, 0])

            save_name = (
                f'{start_time:%Y%m%d_%H%M%S}_{end_time:%H%M%S}_RBSP{self.rbsp_id.upper()}'
                f'_FB{self.fb_id}_summary_plot.png'
                )
            plt.savefig(self.save_path / save_name)
            self._clear_plot()
        return

    def _init_plot(self):
        self.n_rows = 3
        self.n_cols = 2
        self.fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        spec = gridspec.GridSpec(nrows=3, ncols=2, figure=self.fig)
        self.ax = np.zeros((self.n_rows, self.n_cols), dtype=object)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.ax[i,j] = self.fig.add_subplot(spec[i, j])
        return

    def _clear_plot(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.ax[i,j].clear()
        return
        
if __name__ == '__main__':
    fb_id = 3
    rbsp_id = 'A'
    file_name = f'FU{fb_id}_RBSP{rbsp_id.upper()}_conjunctions_dL10_dMLT10_final_hr.csv'

    s = Summary(fb_id, rbsp_id, file_name)
    s.loop()