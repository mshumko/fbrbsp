import json
import dateutil.parser

import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import sampex
import cdflib

import fbrbsp
import fbrbsp.load.utils as utils

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
    time_range: list[str, datetime.datetime, or pd.Timestamp]
        A list, array, or tuple defining the date and time bounds to load.

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
    >>> emfisis = Spec('A', 'WFR', ('2016-01-20T19:00', '2016-01-20T20:00'))
    >>> emfisis.load()
    >>> p, ax = emfisis.spectrum(
    >>>     pcolormesh_kwargs={'norm':matplotlib.colors.LogNorm(vmin=1E-8, vmax=1E-4)}
    >>>     )
    >>> plt.colorbar(p)
    >>> ax.set_yscale('log')
    >>> ax.set_ylim(
    >>>     np.min(emfisis.WFR_frequencies), 
    >>>     np.max(emfisis.WFR_frequencies)
    >>>     )
    >>> plt.show()
    """
    def __init__(self, sc_id, inst, time_range) -> None:
        self.sc_id = sc_id.lower()
        self.inst = inst.lower()
        self.time_range = time_range
        assert self.inst in ['wfr', 'hfr']
        if self.inst != 'wfr':
            raise NotImplementedError(f'{self.inst} is not implemented.')
        self.time_range = utils.validate_time_range(time_range)
        self.spectrum_keys = ['BuBu', 'BvBv', 'BwBw', 'EuEu', 'EvEv', 'EwEw']
        return

    def load(self, missing_ok=True):
        """
        Searches for and loads the L2 EMFISIS spectral-matrix-diagonal-merged 
        data into memory.
        """
        file_dates = utils.get_filename_times(self.time_range, dt='days')
        self.data = {key:np.zeros((0, 65)) for key in self.spectrum_keys}
        self.data['epoch'] = np.array([])

        for file_date in file_dates:
            try:
                file_path = self._find_file(file_date)
            except FileNotFoundError as err:
                if missing_ok and ('EMFISIS files found locally and online' in str(err)):
                    continue
                else:
                    raise
            _cdf = cdflib.CDF(file_path)
            self.data['epoch'] = np.concatenate((
                self.data['epoch'],
                np.array(cdflib.cdfepoch.to_datetime(_cdf['epoch']))
                )) 
            for key in self.spectrum_keys:
                self.data[key] = np.vstack((self.data[key], _cdf[key]))

            if not hasattr(self, 'WFR_frequencies'):
                self.WFR_frequencies = _cdf['WFR_frequencies'].flatten()

        idt = np.where(
            (self.data['epoch'] > self.time_range[0]) & (self.data['epoch'] <= self.time_range[1])
            )[0]
        self.data['epoch'] = self.data['epoch'][idt]
        for key in self.spectrum_keys:
            self.data[key] = self.data[key][idt, :]
        return self.data

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
            fig, ax = plt.subplots()
        else:
            ax = ax

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {
                'norm':matplotlib.colors.LogNorm()
            }

        p = ax.pcolormesh(self['epoch'], self.WFR_frequencies, 
            self[component].T, shading='auto', **pcolormesh_kwargs)

        if fce:
            mag_data = Mag(self.sc_id, self.time_range)
            mag_data.load()
            _fce = mag_data.fce()
            for scaling, ls in zip([1, 0.5, 0.1], ['-', '--', ':']):
                ax.plot(mag_data['Epoch'][::100], scaling*_fce[::100], c='w', ls=ls)
        return p, ax

    def _find_file(self, file_date):
        _file_match = (f'rbsp-{self.sc_id.lower()}_{self.inst.lower()}-'
            f'spectral-matrix-diagonal-merged_emfisis-l2_{file_date:%Y%m%d}_v*.cdf')
        local_files = list(fbrbsp.config["rbsp_data_dir"].rglob(_file_match))

        if len(local_files) == 1:
            self.file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            url = (base_data_url + 
                f'rbsp{self.sc_id.lower()}/l2/emfisis/wfr/'+
                f'spectral-matrix-diagonal-merged/{file_date.year}/'
                )
            downloader = sampex.Downloader(
                url,
                download_dir=fbrbsp.config["rbsp_data_dir"] / f'rbsp_{self.sc_id}' / \
                    'emfisis' / self.inst.lower()
                )
            matched_downloaders = downloader.ls(match=_file_match)
            self.file_path = matched_downloaders[0].download() 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} RBSP-{self.sc_id.upper()} EMFISIS files '
                f'found locally and online that match {_file_match}.'
                )
        return self.file_path

    def __getitem__(self, _slice):
        """
        Access the cdf variables using keys (i.e., ['key']).
        """
        if isinstance(_slice, str):
            if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
                return self.data['epoch']
            elif _slice.lower() in [key.lower() for key in self.spectrum_keys]:
                return self.data[_slice]
            else:
                raise IndexError(f'{_slice} is not in the EMFISIS spectrum data.')
        else:
            raise IndexError(f'Only slicing with integer keys is supported.')


class Mag:
    """
    Load a day of EMFISIS L2 magnetometer data from the computer or the internet.

    Parameters
    ----------
    sc_id: str
        The spacecraft id, can be either A or B, case insensitive.
    time_range: list[str, datetime.datetime, or pd.Timestamp]
        A list, array, or tuple defining the date and time bounds to load.

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
    >>> mag = Mag('A', ('2016-01-20T19:00', '2016-01-20T21:00'))
    >>> mag.load()
    >>> fce = mag.fce()
    """
    def __init__(self, sc_id, time_range) -> None:
        self.sc_id = sc_id.lower()
        self.time_range = utils.validate_time_range(time_range)
        return

    def load(self):
        """
        Searches for and loads the Mag data into memory.
        """
        file_dates = utils.get_filename_times(self.time_range, dt='days')
        self.data = {key:np.array([]) for key in ['Magnitude', 'epoch']}

        for file_date in file_dates:
            file_path = self._find_file(file_date)
            _cdf = cdflib.CDF(file_path)
            self.data['epoch'] = np.concatenate((
                self.data['epoch'],
                np.array(cdflib.cdfepoch.to_datetime(_cdf['epoch']))
                ))
            self.data['Magnitude'] = np.concatenate((
                self.data['Magnitude'], _cdf['Magnitude']
                ))
        
        idt = np.where(
            (self.data['epoch'] > self.time_range[0]) & (self.data['epoch'] <= self.time_range[1])
            )[0]
        self.data['epoch'] = self.data['epoch'][idt]
        self.data['Magnitude'] = self.data['Magnitude'][idt]
        return self.data

    def fce(self):
        """
        Calculate the electron gyrofrequency.
        """
        _fce = 1E-9*self['Magnitude']*np.abs(q_e)/(2*np.pi*m_e)
        return _fce

    def _find_file(self, file_date):
        _file_match = (
            f'rbsp-{self.sc_id.lower()}_magnetometer'
            f'_uvw_emfisis-l2_{file_date:%Y%m%d}_v*.cdf'
            )
        local_files = list(fbrbsp.config["rbsp_data_dir"].rglob(_file_match))

        if len(local_files) == 1:
            file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            url = (base_data_url + 
                f'rbsp{self.sc_id.lower()}/l2/emfisis/magnetometer/'
                f'uvw/{file_date.year}/'
                )
            downloader = sampex.Downloader(
                url,
                download_dir=fbrbsp.config["rbsp_data_dir"] / f'rbsp_{self.sc_id}' / 'emfisis' / 'mag'
                )
            matched_downloaders = downloader.ls(match=_file_match)
            file_path = matched_downloaders[0].download() 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} RBSP-{self.sc_id.upper()} EMFISIS files '
                f'found locally and online that match {_file_match}.'
                )
        return file_path

    def __getitem__(self, _slice):
        """
        Access the cdf variables using keys (i.e., ['key']).
        """
        if isinstance(_slice, str):
            if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
                return self.data['epoch']
            elif _slice.lower() == 'magnitude':
                return self.data['Magnitude']
            else:
                raise IndexError(f'{_slice} is not in the EMFISIS Mag data.')
        else:
            raise IndexError(f'Only slicing with integer keys is supported.')

class Burst:
    """
    Loads and plots EMFISIS L2 waveform-continuous-burst data.

    Parameters
    ----------
    sc_id: str
        The spacecraft id, can be either A or B, case insensitive.
    inst: str
        Select between the wfr or hfr instruments.
    time_range: datetime.datetime, pd.Timestamp
        The time range to load the data.

    Methods
    -------
    load()
        Searches for and loads the L2 EMFISIS spectral-matrix-diagonal-merged 
        data into memory.
    spectrum()

    Example
    -------
    >>> # Replicate Fig. 2e from Breneman et al., (2017) 
    >>> # https://doi.org/10.1002/2017GL075001
    >>> emfisis = Burst('A', 'WFR', ('2016-01-20T19:00', '2016-01-20T20:00'))
    >>> emfisis.load()
    """
    def __init__(self, sc_id, inst, time_range) -> None:
        self.sc_id = sc_id.lower()
        self.inst = inst.lower()
        assert self.inst in ['wfr', 'hfr']
        if self.inst != 'wfr':
            raise NotImplementedError(f'{self.inst} is not implemented.')
        self.time_range = utils.validate_time_range(time_range)
        return

    def load(self):
        """
        Searches for and loads the L2 EMFISIS waveform-continuous-burst 
        data into memory.
        """
        file_dates = utils.get_filename_times(self.time_range)
        self.data = {'epoch':np.array([], dtype=object)}

        for file_date in file_dates:
            file_path = self._find_file(file_date)
            self.data[file_date] = cdflib.CDF(file_path)
            self.data['epoch_start'] = np.append(
                self.data['epoch_start'],
                np.array(cdflib.cdfepoch.to_datetime(self.data[file_date]['epoch']))
            )
            # self.epoch = pd.Timestamp(self._burst_start[0]) + \
            #     pd.to_timedelta(self.cdf['timeOffsets'], unit='nanosecond')
        return self.cdf

    def __getitem__(self, _slice):
        """
        Access the cdf variables using keys (i.e., ['key']).
        """
        if isinstance(_slice, str):
            # TODO: Finish this.
            if "epoch_start" in _slice.lower():
                # Start of each 6-second interval
                idt = np.where(
                    (self.data['epoch_start'] > self.time_range[0]) & 
                    (self.data['epoch_start'] < self.time_range[1])
                    )[0]
                return self.data['epoch_start'][idt]
            elif 'epoch' in _slice.lower():
                # Calculate all epochs (time intensive for long durations).
                _epoch_dict = {}
                for _epoch_start in self.data['epoch_start']:
                    self.epoch = pd.Timestamp(_epoch_start) + \
                        pd.to_timedelta(self.cdf['timeOffsets'], unit='nanosecond')
            else:
                try:
                    return self.data[_slice].to_numpy()
                except KeyError as err:
                    raise KeyError(f'{_slice} slice is unrecognized. Try one'
                        f' of these: {self.data.columns.to_numpy()}')
        else:
            raise KeyError(f'Slice must be str, not {type(_slice)}')

    def _find_file(self, file_date):
        _file_match = (
                f'rbsp-{self.sc_id.lower()}_{self.inst.lower()}-'
                f'waveform-continuous-burst_emfisis-l2_'
                f'{file_date.strftime("%Y%m%dt%H")}_v*.cdf'
                )
        local_files = list(fbrbsp.config["rbsp_data_dir"].rglob(_file_match))

        if len(local_files) == 1:
            self.file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            url = (base_data_url + 
                f'rbsp{self.sc_id.lower()}/l2/emfisis/wfr/'+
                f'waveform-continuous-burst/{file_date.year}/'
                )
            downloader = sampex.Downloader(
                url,
                download_dir=fbrbsp.config["rbsp_data_dir"] / f'rbsp_{self.sc_id}' / 'emfisis' / self.inst
                )
            matched_downloaders = downloader.ls(match=_file_match)
            self.file_path = matched_downloaders[0].download() 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} RBSP-{self.sc_id.upper()} EMFISIS files '
                f'found locally and online that match {self._file_match}.'
                )
        return self.file_path


if __name__ == '__main__':
    # burst = Burst('A', 'WFR', ('2016-01-20T19:00', '2016-01-20T20:30'))
    # burst.load()
    # print(burst['epoch'])
    # pass

    # Replicate Fig. 2c from Breneman et al., (2017) 
    # https://doi.org/10.1002/2017GL075001
    emfisis = Spec('A', 'WFR', ('2016-01-20T19:00', '2016-01-20T20:00'))
    emfisis.load()
    p, ax = emfisis.spectrum(
        pcolormesh_kwargs={'norm':matplotlib.colors.LogNorm(vmin=1E-8, vmax=1E-4)}
        )
    plt.colorbar(p)
    ax.set_yscale('log')
    ax.set_ylim(
        np.min(emfisis.WFR_frequencies), 
        np.max(emfisis.WFR_frequencies)
        )
    plt.show()