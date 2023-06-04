import pathlib
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

import fbrbsp
import fbrbsp.dmlt
from fbrbsp.dial import Dial
from fbrbsp.load.firebird import Hires
from fbrbsp.load.rbsp_magephem import MagEphem


class Conjunction_Dist:
    def __init__(self) -> None:
        self.L = np.array([])
        self.MLT = np.array([])
        self.minMLT = np.array([])
        return
    
    def loop(self):
        for self.fb_id in [3, 4]:
            for self.rbsp_id in ['A', 'B']:
                file_name = f'FU{self.fb_id}_RBSP{self.rbsp_id.upper()}_conjunctions_dL10_dMLT10_final_hr.csv'
                catalog_path = fbrbsp.config['here'].parent / 'data' / file_name
                self.catalog = pd.read_csv(catalog_path)
                self.catalog['startTime'] = pd.to_datetime(self.catalog['startTime'])
                self.catalog['endTime'] = pd.to_datetime(self.catalog['endTime'])

                self._process_file()

    def _process_file(self):
        current_date = datetime.min.date()

        for _, row in self.catalog.iterrows():
            time_range = (row['startTime'],row['endTime'])
            if current_date != row['startTime'].date():
                self.hr = Hires(self.fb_id, row['startTime'].date()).load()
                self.rbsp = MagEphem(self.rbsp_id, 't89d', time_range).load()
                self.rbsp = self.rbsp[list(self.rbsp.keys())[0]]
                current_date = row['startTime'].date()

            L, MLT, minMLT = self._dmlt_crossing(time_range)
            self.L = np.append(self.L, None)
            self.MLT = np.append(self.MLT, None)
            self.minMLT = np.append(self.minMLT, None)
        return
    
    def _dmlt_crossing(self, time_range):
        """
        The L, MLT, and minimum dMLT when the L-shells cross.
        """
        # time_range = [t.to_pydatetime() for t in time_range]
        rb_L, rb_MLT = self._rbsp_filtered_magephem(time_range)
        median_rb_L = np.nanmedian(rb_L)
        hr_idx = np.where(
            (self.hr['Time']>time_range[0]) & 
            (self.hr['Time']<=time_range[1])
            )[0]
        _hr_time = self.hr['Time'][hr_idx]
        _hr_L = self.hr['McIlwainL'][hr_idx]

        try:
            idx = np.nanargmin(np.abs(median_rb_L-_hr_L))
        except ValueError as err:
            if 'All-NaN slice encountered' in str(err):
                return
            else:
                raise
        dMLT = fbrbsp.dmlt.dmlt(np.nanmedian(rb_MLT), self.hr['MLT'][idx])
        return median_rb_L, np.nanmedian(rb_MLT), dMLT

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5,5))
        _dial = Dial(ax, None, None, None)
        _dial.L_labels = [2,4,6,8]
        ax.grid(False) 
        _dial.draw_earth()
        _dial._plot_params()
        s = ax.scatter(self.MLT, self.L, c=self.minMLT)
        plt.colorbar(s, label=f'Minimum ${{\Delta}}$MLT')
        ax.set_title('FIREBIRD-II/RBSP Conjunctions')
        return
    
    def _rbsp_filtered_magephem(self, time_range):
        rb_idx = np.where(
            (self.rbsp['epoch']>time_range[0]) & 
            (self.rbsp['epoch']<=time_range[1])
            )[0]
        ida = np.where(self.rbsp_xlabels['L'].upper() == self.rbsp['L_Label'])[0]
        rb_mlt = self.rbsp[self.rbsp_xlabels['MLT']][rb_idx]
        rb_L = self.rbsp['L'][rb_idx, ida]
        return rb_L, rb_mlt

 
if __name__ == '__main__':
    c = Conjunction_Dist()
    # try:
    c.loop()
    # finally:
    c.plot()
    plt.tight_layout()
    plt.show()