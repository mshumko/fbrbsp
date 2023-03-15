
import dateutil.parser

import pymc3 as pm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.stats

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
        self.center_energy, self.energy_range, self.energy_range_array = super().get_energy_channels()
        return
    
    def plot(self, choose_n_samples=500):
        """
        Parameters
        ----------
        choose_n_samples: int
            How many lines to plot

        Returns
        -------
        list[plt.Axes]
            The list of subplots.
        """
        self.ax = super().plot(self._time)
    
        if hasattr(self, 'trace'):
            energies = np.linspace(200, 1000)
            # energies = np.linspace(
            #     self.center_energy[0] - self.xerrs[0,0], self.center_energy[-1] + self.xerrs[0,-1]
            #     )

            idx = np.random.choice(np.arange(len(self.trace['slope'])), choose_n_samples, replace=False)
            lines = np.nan*np.zeros((energies.shape[0], choose_n_samples))
            for i, idx_i in enumerate(idx):
                lines[:, i] = self.trace['intercept'][idx_i] + energies*self.trace['slope'][idx_i]
                # self.ax.plot(energies, lines[:, i], c='grey', alpha=0.2)
            lower_boundary = np.quantile(lines, 0.025, axis=1)
            upper_boundary = np.quantile(lines, 0.975, axis=1)
            self.ax[-1].fill_between(energies, lower_boundary, upper_boundary, color='grey', alpha=0.5)
            self.ax[-1].plot(energies, self.trace['intercept'].mean() + energies*self.trace['slope'].mean(), 'k--')

            quantiles = np.round(np.quantile(self.trace["slope"], [0.025, 0.975]), 2)
            linear_fit_str = (f'slope = {round(self.trace["slope"].mean(), 2)} [ms/keV]'
                f' | CI = [{quantiles[0]}, {quantiles[1]}]')
            self.ax[-1].text(0.99, 0, linear_fit_str, transform=self.ax[-1].transAxes, 
                        va='bottom', ha='right', color='k', fontsize=13)
                
        if self.energy_dist is not None:
            _xerrs = self.xerrs
        else:
            _xerrs = None
        self.ax[-1].errorbar(self.center_energy, self.t0_diff_ms, c='k', marker='.', 
            yerr=self.yerrs, xerr=_xerrs, capsize=2, ls='None')
        max_abs_lim = 1.1*np.max(np.abs(self.ax[-1].get_ylim()))
        self.ax[-1].set_ylim(-max_abs_lim, max_abs_lim)
        self.ax[-1].axhline(c='k', ls='-')
        self.ax[-1].set(xlabel='Energy [keV]', ylabel=f'Peak time lag [ms]\n$(ch_{{n}}-ch_{{0}})$')

        locator=matplotlib.ticker.FixedLocator(np.linspace(-max_abs_lim, max_abs_lim, num=5))
        self.ax[-1].yaxis.set_major_locator(locator)
        return self.ax
    
    def fit_dispersion(self, samples=10_000, tune=20_000, energy_dist=None):
        """
        The Bayes errors-in-variables (energy) model.

        See https://discourse.pymc.io/t/errors-in-variables-model-in-pymc3/3519
        and https://arxiv.org/pdf/1906.03989.pdf.

        Parameters
        ----------
        samples: int
            The number of samples in the trace.
        tune: int
            The number of tuning (burn-in) steps.
        energy_dist: str
            The energy channel uncertainty model. Can be None for no uncertainty,
            'uniform', or 'exp'. 
            - None: All counts arrived with a single energy at the center energy
            of each energy channel.
            - 'uniform': all counts are uniformly distributed inside each channel's
            energy bounds.
            - 'exp': the counts are exponentially distributed in each energy 
            channel with the exponential decay parameter (E_0) calculated by
            fitting an exponential to the Col_flux variable.
        """
        if energy_dist is None:
            self.energy_dist = None
        else:
            self.energy_dist = energy_dist.lower()

        with pm.Model() as model:
            # Parameters we ultimately care about
            intercept = pm.Normal("intercept", 0, sigma=10)
            slope = pm.Normal("slope", 0, sigma=1)

            # call energy.random(size=...) to generate random values and confirm
            # that they are correctly generated. See 
            # https://www.pymc.io/projects/docs/en/v3/Probability_Distributions.html#using-pymc-distributions-without-a-model
            if self.energy_dist == 'uniform': 
                energy = pm.Uniform('energy',
                                lower=self.energy_range_array[:, 0], 
                                upper=self.energy_range_array[:, 1],
                                shape=self.energy_range_array.shape[0]
                                )
            elif self.energy_dist == 'exp':
                _energy_spec = self.fit_energy_spectrum()
                # First set of parameters for the Bound, and second set for Exponential.
                energy = pm.Bound(pm.Exponential,
                    lower=self.energy_range_array[:, 0], 
                    upper=self.energy_range_array[:, 1]
                    )('energy', lam=1/_energy_spec['E0'], 
                    shape=self.energy_range_array.shape[0])
            elif self.energy_dist is None:
                energy = self.center_energy
            else:
                raise NotImplementedError

            likelihood = pm.Normal("y", 
                mu=intercept + slope * energy, 
                sigma=self.cadence_ms/2,
                observed=self.t0_diff_ms)                                
            # cores=1 due to a multiprocessing bug in Windows's pymc3. 
            # See this discussion: https://discourse.pymc.io/t/error-during-run-sampling-method/2522. 
            self.trace = pm.sample(samples, cores=1, tune=tune)
        return
    
    def fit_energy_spectrum(self, validate=False):
        """
        Calculate the energy spectrum of the HiRes data assuming an exponential spectrum model.
        """
        fit_interval_idx = np.where(
                (self.hr['Time'] > self.microburst_info['Time']-self.fit_interval_s/2) &
                (self.hr['Time'] <= self.microburst_info['Time']+self.fit_interval_s/2)
            )[0]
        integrated_flux = np.sum(self.hr['Col_flux'][fit_interval_idx], axis=0)[self.channels]

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            self.center_energy, y=np.log(integrated_flux)
            )
        _spectrum = {'J0': np.exp(intercept), 'E0':(-1/slope)}

        if validate:
            energies = np.linspace(self.center_energy[0], self.center_energy[-1])
            plt.step(self.center_energy, integrated_flux)
            plt.plot(energies, _spectrum['J0']*np.exp(-energies/_spectrum['E0']))
            plt.yscale('log')
            plt.show()
        return _spectrum

    def get_dispersion(self):
        """
        
        """
        return super()._get_dispersion()


if __name__ == '__main__':
    plot_window_s=1

    ## Best positive dispersion event so far
    # time = '2015-08-27T12:40:37'
    # channels = np.arange(4)

    ## DAPPER saturation 
    time = '2015-02-02T06:12:26.310000'
    channels = np.arange(5)


    fb_id = 3
    catalog_version=5
    fit_interval_s = 0.3

    model = Bayes_Fit(fb_id, channels=channels, catalog_version=catalog_version, 
                    fit_interval_s=fit_interval_s, plot_window_s=plot_window_s, 
                    full_ylabels=True)
    model.load(time)
    model.get_dispersion()
    model.fit_dispersion(energy_dist='exp')
    pass
    ax = model.plot()
    ax[-1].set_xlim(200, 1000)
    ax[-1].set_ylim(-80, 80)
    loc = matplotlib.ticker.MaxNLocator(7) # this locator puts ticks at regular intervals
    ax[-1].yaxis.set_major_locator(loc)
    plt.show()
    # print(f'{model.t0_diff_ms=}')
    # print(f'{model.center_energy=}')
    pass