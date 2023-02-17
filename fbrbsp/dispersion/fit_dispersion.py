
import dateutil.parser

import pymc3 as pm
from pymc3 import HalfCauchy, Model, Normal, sample
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import xarray as xr

import plot_dispersion
import fbrbsp
import fbrbsp.load.firebird

print(f"Running on PyMC v{pm.__version__}")

class Bayes_Fit(plot_dispersion.Dispersion):
    def __init__(self, fb_id:int, channels:list=np.arange(6), 
                 catalog_version:int=5, fit_interval_s:float=0.3, 
                 plot_window_s:float=1, full_ylabels:bool=True,
                 annotate_fit:bool=False) -> None:
        """
        Calculate the energy dispersion using plot_dispersion.Dispersion
        and fit it using pymc3.

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
        full_ylabels: bool
            Draw energy channel number or the keV energy range.
        annotate_fit: bool
            Add an "Energy" arrow pointing up.
        """
        super().__init__(fb_id, channels=channels, 
                 catalog_version=catalog_version, fit_interval_s=fit_interval_s, 
                 plot_window_s=plot_window_s, full_ylabels=full_ylabels,
                 annotate_fit=annotate_fit)
        return
    
    def load(self, time):
        """
        Loads the HiRes data into self.hr class attribute.

        It also creates a self.cadence_s, self.cadence_ms, self.center_energy, 
        and self.energy_range attributes
        """
        self._time = time
        if isinstance(self._time, str):
            self._time = dateutil.parser.parse(self._time)
        idt = np.argmin(np.abs(self.catalog['Time']-self._time))
        self.microburst_info = self.catalog.loc[idt, :]
        dt = np.abs((self.microburst_info['Time']-self._time).total_seconds())
        if dt > 1:
            raise ValueError(f'The desired microburst plot time is {dt} '
                             f'seconds away from the closest microburst '
                             f'observed at {self.microburst_info["Time"]}')
        self.hr = fbrbsp.load.firebird.Hires(self.fb_id, time).load()
        self.cadence_s = float(self.hr.attrs["CADENCE"])
        self.cadence_ms = 1000*self.cadence_s
        self.center_energy, self.energy_range = self.get_energy_channels()
        return
    
    def plot(self, ax=None, n_samples=500):

        if ax is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = ax

        if hasattr(self, 'trace'):
            energies = np.linspace(
                self.center_energy[0] - self.xerrs[0,0], self.center_energy[-1] + self.xerrs[0,-1]
                )
            self.ax.plot(energies, model.trace['intercept'].mean() + energies*model.trace['slope'].mean(), 'k:')

            idx = np.random.choice(np.arange(len(model.trace['slope'])), n_samples, replace=False)
            for idx_i in idx:
                self.ax.plot(energies, model.trace['intercept'][idx_i] + energies*model.trace['slope'][idx_i], 
                            c='grey', alpha=0.2)
                
        self.ax.errorbar(self.center_energy, self.t0_diff_ms, c='k', marker='.', 
            yerr=self.yerrs, xerr=self.xerrs, capsize=2, ls='None')
        max_abs_lim = 1.1*np.max(np.abs(self.ax.get_ylim()))
        self.ax.set_ylim(-max_abs_lim, max_abs_lim)
        self.ax.axhline(c='k', ls='--')
        self.ax.set(xlabel='Energy [keV]', ylabel='Peak time delay [ms]\n(ch[N]-ch[0])')

        locator=matplotlib.ticker.FixedLocator(np.linspace(-max_abs_lim, max_abs_lim, num=5))
        self.ax.yaxis.set_major_locator(locator)
        return
    
    def fit(self):
        """
        
        """
        with Model() as model:
            intercept = Normal("intercept", 0, sigma=100)
            slope = Normal("slope", 0, sigma=100)

            likelihood = Normal("y", mu=intercept + slope * self.center_energy, 
                                sigma=self.cadence_ms, observed=self.t0_diff_ms)
            # cores=1 due to a multiprocessing bug in Windows's pymc3. 
            # See this discussion: https://discourse.pymc.io/t/error-during-run-sampling-method/2522. 
            self.trace = sample(10_000, cores=1, tune=10_000)
        return

    def get_dispersion(self):
        """
        
        """
        return super()._get_dispersion()


plot_window_s=1

## Best positive dispersion event so far
time = '2015-08-27T12:40:37'
channels = np.arange(4)


fb_id = 3
catalog_version=5
fit_interval_s = 0.3

model = Bayes_Fit(fb_id, channels=channels, catalog_version=catalog_version, 
                fit_interval_s=fit_interval_s, plot_window_s=plot_window_s, 
                full_ylabels=True)
model.load(time)
model.get_dispersion()
model.fit()
pass
model.plot()
plt.show()
# print(f'{model.t0_diff_ms=}')
# print(f'{model.center_energy=}')