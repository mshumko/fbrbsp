import pathlib
from datetime import datetime, timedelta

import numpy as np
import scipy.optimize
import pandas as pd

import fbrbsp
import fbrbsp.load.firebird


class Duration:
    def __init__(self, fb_id, catalog_version, detrend=True, max_cadence=18.75, channel=0) -> None:
        self.fb_id = fb_id  
        self.microburst_name = f'FU{fb_id}_microburst_catalog_{str(catalog_version).zfill(2)}.csv'
        self.microburst_path = fbrbsp.config['here'].parent / 'data' / self.microburst_name
        self.detrend = detrend
        self.max_cadence = max_cadence
        self.channel = channel

        self._load_catalog()
        self._load_campaign_dates()

        return

    def loop(self,):

        current_date = datetime.min

        for i, row in self.microbursts.iterrows():
            if self._get_cadence(row['Time']) > self.max_cadence:
                continue
            if current_date != row['Time'].date():
                self.hr = fbrbsp.load.firebird.Hires(self.fb_id, row['Time'].date()).load()
                current_date = row['Time'].date()

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

if __name__ == "__main__":
    d = Duration(3, 5)
    d.loop()