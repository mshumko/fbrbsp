import pathlib
from datetime import datetime, timedelta
import warnings
import time

import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates
import sklearn.metrics

import fbrbsp
import fbrbsp.load.firebird


class Duration:
    def __init__(self, fb_id, catalog_version, detrend=True, max_cadence=18.75, 
                channel=0, validation_plots=False) -> None:
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

    def loop(self):
        """
        Loop over and fit each microburst that was detected when the HiRes cadence was faster or
        equal to self.max_cadence.
        """
        start_time = time.time()
        self.fit_param_names = ['r2', 'adj_r2', 'A', 't0', 'fwhm']
        if self.detrend:
            self.fit_param_names.extend(['y-int', 'slope'])
        self.microbursts[self.fit_param_names] = np.nan
        current_date = datetime.min

        for i, row in self.microbursts.iterrows():
            print(f"Processing {row['Time']} microburst ({i}/{self.microbursts.shape[0]}).")
            if self._get_cadence(row['Time']) > self.max_cadence:
                continue
            if current_date != row['Time'].date():
                self.hr = fbrbsp.load.firebird.Hires(self.fb_id, row['Time'].date()).load()
                current_date = row['Time'].date()

            popt, r2, adj_r2 = self.fit(row)
            if popt is None:
                continue  # Fit failed. Leave the fit columns as nans in self.microburst catalog.
            if self.detrend:
                self.microbursts.loc[i, self.fit_param_names] = [r2, adj_r2, *popt]
            else:
                self.microbursts.loc[i, self.fit_param_names] = [r2, adj_r2, *popt]
            
            if self.validation_plots:
                self._plot_microburst(i, self.microbursts.loc[i, :])

        self.microbursts.to_csv(self.microburst_path, index=False)
        print(f'Microburst fitting completed in {(time.time() - start_time)//60} minutes.')
        return

    def fit(self, row, fit_interval_s=0.3):
        """
        Fit the microburst at time row['Time'] by a Gaussian. Energy channel defined by self.channel.
        """
        idt_peak = np.where(self.hr['Time'] == row['Time'])[0][0]
        t0_peak = self.hr['Time'][idt_peak]
        time_range = [
                self.hr['Time'][idt_peak]-pd.Timedelta(seconds=fit_interval_s/2),
                self.hr['Time'][idt_peak]+pd.Timedelta(seconds=fit_interval_s/2)
                ]
        idt = np.where(
            (self.hr['Time'] > time_range[0]) &
            (self.hr['Time'] < time_range[1])
            )[0]

        if self.detrend:
            p0 = [
                self.hr['Col_counts'][idt_peak, self.channel],   # gauss amplitude 
                t0_peak,         # gauss center time
                0.1,    # 2x gaus std.
                np.median(self.hr['Col_counts'][idt, self.channel]), # y-intercept
                0           # Slope
                ]
        else:
            p0 = [self.hr['Col_counts'][idt_peak, self.channel], t0_peak, 0.1]

        with warnings.catch_warnings(record=True) as w:
            try:
                popt, pcov, r2, adj_r2 = self._fit_gaus(idt, p0)
            except RuntimeError as err:
                if ('Optimal parameters not found: Number of calls '
                    'to function has reached maxfev') in str(err):
                    return (None, None, None)
                else:
                    raise
            if len(w):  # only print if warning emitted.
                print(w[0].message, '\n', p0, popt)
        return popt, r2, adj_r2

    def _fit_gaus(self, idt, p0):
        """
        Fits a gaussian shape with an optional linear detrend term.
        """
        x_data = self.hr['Time'][idt]
        current_date = x_data[0].floor('d')
        x_data_seconds = (x_data-current_date).total_seconds()
        y_data = self.hr['Col_counts'][idt, self.channel]

        if len(x_data) < len(p0):
            raise ValueError('Not enough data points to fit. Increase the '
                            'time_range or self.width_multiplier')

        p0[0] *= 2
        p0[1] = (p0[1] - current_date).total_seconds()
        p0[2] = p0[2]/2 # Convert the microburst width guess to ~std.

        popt, pcov = scipy.optimize.curve_fit(Duration.gaus_lin_function, 
                                                x_data_seconds, y_data, p0=p0, maxfev=5000)
        popt_np = -1*np.ones(len(popt), dtype=object)
        popt_np[0] = popt[0]
        popt_np[1] = current_date + pd.Timedelta(seconds=float(popt[1]))
        popt_np[2] = np.abs((2*np.sqrt(2*np.log(2)))*popt[2])
        if len(popt) == 5:
            # If superposed a Gaussian on a linear trend...
            popt_np[3:] = popt[3:]

        y_pred = Duration.gaus_lin_function(x_data_seconds, *popt)
        try:
            r2, adj_r2 = self.goodness_of_fit(y_data, y_pred, len(popt))
        except ValueError as err:
            if 'Input contains NaN, infinity or a value too large' in str(err):
                print(f'popt={popt}')
                print(f'y-data={y_data}')
                print(f'y_pred={y_pred}')
            raise
        return popt_np, np.sqrt(np.diag(pcov)), r2, adj_r2

    @staticmethod
    def gaus_lin_function(t, *args):
        """
        Args is an array of either 3 or 5 elements. First three elements are
        the Guassian amplitude, center time, and width. The last two optional
        elements are the y-intercept and slope for a linear trend. 
        """
        exp_arg = -(t-args[1])**2/(2*args[2]**2)
        y = args[0]*np.exp(exp_arg)

        if len(args) == 5:
            y += args[3] + t*args[4]
        return y

    def goodness_of_fit(self, y_true, y_pred, n_params):
        """
        Method to calculate the R^2 coefficient of determination
        and the adjusted R^2 coefficient given the number
        of fit parameters n_params.
        """
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        n = len(y_true)
        adj_r2 = 1 - (1-r2)*(n-1)/(n-1-n_params)
        return r2, adj_r2

    def _load_catalog(self):
        """
        Load a microburst catalog
        """
        self.microbursts = pd.read_csv(self.microburst_path)
        self.microbursts['Time'] = pd.to_datetime(self.microbursts['Time'])
        return self.microbursts

    def _get_cadence(self, time):
        """
        Gets the cadence at the time the microburst was observed. 
        """
        df = self.campaign.loc[:, ['HiRes Cadence', 'Start Date', 'End Date']]
        for _, (cadence, start_date, end_date) in df.iterrows():
             if (pd.Timestamp(time.date()) >= start_date) & (pd.Timestamp(time.date()) <= end_date):
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

    def _plot_microburst(self, i, row, plot_window_s=2):
        """
        Make validation plots of each microburst.
        """
        _, ax = plt.subplots()
        # Plot the data
        index = row['Time']
        dt = pd.Timedelta(seconds=plot_window_s/2)
        time_range = (index-dt, index+dt)

        idt = np.where(
            (self.hr['Time'] > time_range[0]) &
            (self.hr['Time'] < time_range[1])
            )[0]
        idt_peak = np.where(self.hr['Time'] == index)[0]
        ax.plot(self.hr['Time'][idt], self.hr['Col_counts'][idt, self.channel], c='k')
        ax.scatter(self.hr['Time'][idt_peak], 
            self.hr['Col_counts'][idt_peak, self.channel], marker='*', s=200, c='r')

        # Plot the fit
        time_array = self.hr['Time'][idt]
        current_date = time_array[0].floor('d')
        x_data_seconds = (time_array-current_date).total_seconds()

        if self.detrend:
            popt = np.nan*np.zeros(5)
            popt[3] = self.microbursts.loc[i, 'y-int']
            popt[4] = self.microbursts.loc[i, 'slope']
        else:
            popt = np.nan*np.zeros(3)
        popt[0] = self.microbursts.loc[i, 'A']
        popt[1] = (self.microbursts.loc[i, 't0'] - current_date).total_seconds()
        popt[2] = self.microbursts.loc[i, 'fwhm']/2.355 # Convert the Gaussian FWHM to std

        gaus_y = Duration.gaus_lin_function(x_data_seconds, *popt)
        ax.plot(time_array, gaus_y, c='r')

        ax.set(
            xlim=time_range, xlabel='Time', 
            ylabel=f'Counts/{1000*float(self.hr.attrs["CADENCE"])} ms',
            title=index.strftime("%Y-%m-%d %H:%M:%S.%f\nmicroburst fit validation")
            )
        s = (
            f'L={round(row["McIlwainL"], 1)}\n'
            f'MLT={round(row["MLT"], 1)}\n'
            f'(lat,lon)=({round(row["Lat"], 1)}, {round(row["Lon"], 1)})\n'
            f"FWHM={round(self.microbursts.loc[i, 'fwhm'], 2)} [s]\n"
            f"R^2 = {round(self.microbursts.loc[i, 'r2'], 2)}\n"
            f"adj_R^2 = {round(self.microbursts.loc[i, 'adj_r2'], 2)}\n"
            )
        ax.text(0.7, 1, s, va='top', transform=ax.transAxes, color='red')
        ax.set_ylim(0, 1.2*np.max(self.hr['Col_counts'][idt, self.channel]))
        locator=matplotlib.ticker.MaxNLocator(nbins=5)
        ax.xaxis.set_major_locator(locator)
        fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(fmt)

        plt.tight_layout()

        save_time = index.strftime("%Y%m%d_%H%M%S_%f")
        save_name = (f'{save_time}_fu{self.fb_id}_microburst_fit.png')
        save_path = pathlib.Path(self.plot_save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        return


if __name__ == "__main__":
    d = Duration(3, 5, validation_plots=False)
    d.loop()