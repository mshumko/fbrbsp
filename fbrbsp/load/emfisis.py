import json
import dateutil.parser

import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import sampex
import cdflib

import fbrbsp

base_data_url = 'https://spdf.gsfc.nasa.gov/pub/data/rbsp/'
q_e = -1.6E-19
m_e = 9.1E-31

class Spec:
    """
    Loads and plots a day of EMFISIS L2 spectrum data.

    Parameters
    ----------
    sc_id: str
        The spacecraft id, can be either A or B, case insensitive.
    inst: str
        Select between the wfr or hfr instruments.
    load_date: datetime.datetime, pd.Timestamp
        The date to load the data.

    Methods
    -------
    load()
        Searches for and loads the L2 EMFISIS spectral-matrix-diagonal-merged 
        data into memory.
    spectrum(component='BuBu', fce=True, ax=None, pcolormesh_kwargs=None)
        Plots a component of the EMFISIS WFR spectrum (controlled via the component 
        kwarg) with optional f_ce lines superposed.

    Example
    -------
    >>> # Replicate Fig. 2c from Breneman et al., (2017) 
    >>> # https://doi.org/10.1002/2017GL075001
    >>> emfisis = Spec('A', 'WFR', '2016-01-20')
    >>> emfisis.load()
    >>> p, ax = emfisis.spectrum(
    >>>     pcolormesh_kwargs={'norm':matplotlib.colors.LogNorm(vmin=1E-8, vmax=1E-4)}
    >>>     )
    >>> plt.colorbar(p)
    >>> ax.set_yscale('log')
    >>> ax.set_xlim(
    >>>     dateutil.parser.parse('2016-01-20T19:00'),
    >>>     dateutil.parser.parse('2016-01-20T20:00')
    >>>     )
    >>> ax.set_ylim(
    >>>     np.min(emfisis.cdf['WFR_frequencies']), 
    >>>     np.max(emfisis.cdf['WFR_frequencies'])
    >>>     )
    >>> plt.show()
    """
    def __init__(self, sc_id, inst, load_date) -> None:
        self.sc_id = sc_id.lower()
        self.inst = inst.lower()
        assert self.inst in ['wfr', 'hfr']
        if self.inst != 'wfr':
            raise NotImplementedError(f'{self.inst} is not implemented.')
        if isinstance(load_date, str):
            self.load_date = dateutil.parser.parse(load_date)
        else:
            self.load_date = load_date
        return

    def load(self):
        """
        Searches for and loads the L2 EMFISIS spectral-matrix-diagonal-merged 
        data into memory.
        """
        self._file_match = (f'rbsp-{self.sc_id.lower()}_{self.inst.lower()}-'
            f'spectral-matrix-diagonal-merged_emfisis-l2_{self.load_date:%Y%m%d}_v*.cdf')
        self.file_path = self._find_file()
        self.cdf = cdflib.CDF(self.file_path)
        self.epoch = np.array(cdflib.cdfepoch.to_datetime(self.cdf['epoch']))
        return self.cdf

    def _find_file(self):
        local_files = list(fbrbsp.config["rbsp_data_dir"].rglob(self._file_match))

        if len(local_files) == 1:
            self.file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            url = (base_data_url + 
                f'rbsp{self.sc_id.lower()}/l2/emfisis/wfr/'+
                f'spectral-matrix-diagonal-merged/{self.load_date.year}/'
                )
            downloader = sampex.Downloader(
                url,
                download_dir=fbrbsp.config["rbsp_data_dir"] / f'rbsp_{self.sc_id}' / 'emfisis'
                )
            matched_downloaders = downloader.ls(match=self._file_match)
            self.file_path = matched_downloaders[0].download() 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} RBSP-{self.sc_id.upper()} EMFISIS files '
                f'found locally and online that match {self._file_match}.'
                )
        return self.file_path

    def spectrum(self, component='BuBu', fce=True, ax=None, pcolormesh_kwargs=None):
        """
        Plots a component of the EMFISIS WFR spectrum (controlled via the component 
        kwarg) with optional f_ce lines superposed.

        Parameters
        ----------
        component: str
            The magnetic or electric field component. Can be one of
            'BuBu', 'BvBv', 'BwBw', 'EuEu', 'EvEv', or 'EwEw'.
        fce: bool
            Plot the f_ce, f_ce/2, and f_ce/10 lines.
        ax: plt.Axes
            A subplot to plot on.
        pcolormesh_kwarg: dict
            The keyword arguments to pass into plt.pcolormesh. By default
            the only setting is a logarithmic color scale.
        """
        if ax is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = ax

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {
                'norm':matplotlib.colors.LogNorm()
            }

        p = self.ax.pcolormesh(self.epoch, self.cdf['WFR_frequencies'].flatten(), 
            self.cdf[component].T, shading='auto', **pcolormesh_kwargs)

        if fce:
            mag_times, _fce = self._calc_fce()
            for multiple in [1, 0.5, 0.1]:
                self.ax.plot(mag_times[::100], multiple*_fce[::100], c='k')
        return p, self.ax

    def _calc_fce(self):
        self.mag = Mag(self.sc_id, self.load_date)
        self.mag.load()
        # 1E-9 Convert to Tesla
        _fce = 1E-9*self.mag.cdf['Magnitude']*np.abs(q_e)/(2*np.pi*m_e)
        return self.mag.epoch, _fce


class Mag:
    """
    Load a day of EMFISIS L2 magnetometer data from the computer or the internet.

    Parameters
    ----------
    sc_id: str
        The spacecraft id, can be either A or B, case insensitive.
    load_date: datetime.datetime, pd.Timestamp
        The date to load the data.

    Methods
    -------
    load()
        Searches for and loads the Mag data into memory.
    fce()
        Calculate the electron gyrofrequency.

    Example
    -------
    >>> # Calculate the electron gyrofrequency during the conjunction analyzed by
    >>> # Breneman et al., (2017) https://doi.org/10.1002/2017GL075001
    >>> q_e = -1.6E-19
    >>> m_e = 9.1E-31
    >>> mag = Mag('A', '2016-01-20')
    >>> mag.load()
    >>> fce = mag.fce()
    """
    def __init__(self, sc_id, load_date) -> None:
        self.sc_id = sc_id.lower()
        if isinstance(load_date, str):
            self.load_date = dateutil.parser.parse(load_date)
        else:
            self.load_date = load_date
        return

    def load(self):
        """
        Searches for and loads the Mag data into memory.
        """
        # rbsp-a_magnetometer_uvw_emfisis-l2_20160101_v1.6.4.cdf
        self._file_match = (
            f'rbsp-{self.sc_id.lower()}_magnetometer'
            f'_uvw_emfisis-l2_{self.load_date:%Y%m%d}_v*.cdf'
            )
        self.file_path = self._find_file()
        self.cdf = cdflib.CDF(self.file_path)
        self.epoch = np.array(cdflib.cdfepoch.to_datetime(self.cdf['epoch']))
        return self.cdf

    def fce(self):
        """
        Calculate the electron gyrofrequency.
        """
        _fce = 1E-9*self.cdf['Magnitude']*np.abs(q_e)/(2*np.pi*m_e)
        return _fce


    def _find_file(self):
        local_files = list(fbrbsp.config["rbsp_data_dir"].rglob(self._file_match))

        if len(local_files) == 1:
            self.file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            url = (base_data_url + 
                f'rbsp{self.sc_id.lower()}/l2/emfisis/magnetometer/'
                f'uvw/{self.load_date.year}/'
                )
            downloader = sampex.Downloader(
                url,
                download_dir=fbrbsp.config["rbsp_data_dir"] / f'rbsp_{self.sc_id}' / 'emfisis'
                )
            matched_downloaders = downloader.ls(match=self._file_match)
            self.file_path = matched_downloaders[0].download() 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} RBSP-{self.sc_id.upper()} EMFISIS files '
                f'found locally and online that match {self._file_match}.'
                )
        return self.file_path
