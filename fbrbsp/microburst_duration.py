import pathlib
from datetime import datetime, timedelta

import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates

import fbrbsp
import fbrbsp.load.firebird


class Duration:
    def __init__(self, fb_id, catalog_version, detrend=True, max_cadence=18.75, 
                channel=0, validation_plots=True) -> None:
        self.fb_id = fb_id  
        self.microburst_name = f'FU{fb_id}_microburst_catalog_{str(catalog_version).zfill(2)}.csv'
        self.microburst_path = fbrbsp.config['here'].parent / 'data' / self.microburst_name
        self.detrend = detrend
        self.max_cadence = max_cadence
        self.channel = channel
        self.validation_plots = validation_plots

        if self.validation_plots:
            self.plot_save_dir = pathlib.Path(fbrbsp.config['here'].parent, 'plots', 
                self.microburst_name.split('.')[0])
            self.plot_save_dir.mkdir(parents=True, exist_ok=True)

        self._load_catalog()
        self._load_campaign_dates()

        return

    def loop(self,):
        """
        Loop over and fit each microburst that was detected when the HiRes cadence was faster or
        equal to self.max_cadence.
        """
        current_date = datetime.min

        for i, row in self.microbursts.iterrows():
            if self._get_cadence(row['Time']) > self.max_cadence:
                continue
            if current_date != row['Time'].date():
                self.hr = fbrbsp.load.firebird.Hires(self.fb_id, row['Time'].date()).load()
                current_date = row['Time'].date()
            
            if self.validation_plots:
                self._plot_microburst(self, row)

        return

    def _load_catalog(self):
        """
        Load a microburst catalog
        """
        self.microbursts = pd.read_csv(self.microburst_path)
        self.microbursts['Time'] = pd.to_datetime(self.microbursts['Time'])
        self.fit_param_names = ['r2', 'adj_r2', 'A', 't0', 'fwhm']
        if self.detrend:
            self.fit_param_names.extend(['y-int', 'slope'])
        self.microbursts[self.fit_param_names] = np.nan
        return self.microbursts

    def _get_cadence(self, time):
        """
        Gets the cadence at the time the microburst was observed. 
        """
        df = self.campaign.loc[:, ['HiRes Cadence', 'Start Date', 'End Date']]
        for i, (cadence, start_date, end_date) in df.iterrows():
             if (time >= start_date) & (time <= end_date):
                return cadence
        raise ValueError(f'Could not find a corresponding HiRes cadence on {time}')

    def _load_campaign_dates(self):
        """
        Load the FIREBIRD-II campaign csv file from the data README. This is necessary to process
        only the microbursts detected during a campaign with a fast enough cadence.
        """
        self.campaign_name = f'fb_campaigns.csv'
        self.campaign_path = fbrbsp.config['here'].parent / 'data' / self.campaign_name
        self.campaign = pd.read_csv(self.campaign_path)
        self.campaign['Start Date'] = pd.to_datetime(self.campaign['Start Date'])
        self.campaign['End Date'] = pd.to_datetime(self.campaign['End Date'])
        self.campaign['HiRes Cadence'] = [float(c.split()[0]) for c in self.campaign['HiRes Cadence']]
        return

    def _plot_microburst(self, row, plot_window_s=2):
        _, ax = plt.subplots()
        index = row['Time']
        dt = pd.Timedelta(seconds=plot_window_s/2)
        time_range = (index-dt, index+dt)

        idt = np.where(
            (self.hr['Time'] > time_range[0]) &
            (self.hr['Time'] < time_range[1])
            )[0]
        idt_peak = np.where(self.hr['Time'] == index)[0]
        ax.plot(self.hr['Time'][idt], self.hr['Col_counts'][idt, 0], c='k')
        ax.scatter(self.hr['Time'][idt_peak], self.hr['Col_counts'][idt_peak, 0], marker='*', s=200, c='r')

        ax.set(
            xlim=time_range, xlabel='Time', 
            ylabel=f'Counts/{1000*float(self.hr.attrs["CADENCE"])} ms',
            title=index.strftime("%Y-%m-%d %H:%M:%S.%f\nmicroburst validation")
            )
        s = (
            f'time_gap={row["time_gap"]}\nsaturated={row["saturated"]}\n'
            f'n_zeros={row["n_zeros"]}\n\n'
            f'L={round(row["McIlwainL"], 1)}\n'
            f'MLT={round(row["MLT"], 1)}\n'
            f'(lat,lon)=({round(row["Lat"], 1)}, {round(row["Lon"], 1)})'
            )
        ax.text(0.7, 1, s, va='top', transform=ax.transAxes, color='red')
        locator=matplotlib.ticker.MaxNLocator(nbins=5)
        ax.xaxis.set_major_locator(locator)
        fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(fmt)

        plt.tight_layout()

        save_time = index.strftime("%Y%m%d_%H%M%S_%f")
        save_name = (f'{save_time}_fu{self.fb_id}_microburst.png')
        save_path = pathlib.Path(self.plot_save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        return


if __name__ == "__main__":
    d = Duration(3, 5)
    d.loop()