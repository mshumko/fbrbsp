import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.colors
import matplotlib.pyplot as plt
import sampex
import cdflib

import fbrbsp
import fbrbsp.load.utils as utils
from fbrbsp.load.emfisis import Mag

base_data_url = 'https://spdf.gsfc.nasa.gov/pub/data/rbsp/'
q_e = -1.6E-19
m_e = 9.1E-31

class Burst1:
    """
    Loads and plots a day of EFW burst 1 data (scientist in the loop).

    Parameters
    ----------
    sc_id: str
        The spacecraft id, can be either A or B, case insensitive.
    time_range: list[str, datetime.datetime, or pd.Timestamp]
        A list, array, or tuple defining the date and time bounds to load.

    Methods
    -------
    load()
        Searches for and loads the L1 EFW burst 1 data into memory.
    spectrum(component='BuBu', fce=True, ax=None, pcolormesh_kwargs=None)
        Plots a component of the EFW burst 1 spectrum (controlled via the component 
        kwarg) with optional f_ce lines superposed.

    Example
    -------
    
    """
    def __init__(self, sc_id, time_range) -> None:
        self.sc_id = sc_id.lower()
        self.time_range = time_range
        self.time_range = utils.validate_time_range(time_range)
        self.sample_keys = ['BuSamples', 'BvSamples', 'BwSamples', 
            'EuSamples', 'EvSamples', 'EwSamples']
        return

    def load(self, missing_ok=True):
        """
        Searches for and loads the L1 EFW burst 1 data into memory.
        """
        file_dates = utils.get_filename_times(self.time_range, dt='days')
        self.cdf_handles = {}

        for file_date in file_dates:
            try:
                file_path = self._find_file(file_date)
            except FileNotFoundError as err:
                if missing_ok and ('EFW mscb1 (burst 1) files found locally and online' in str(err)):
                    continue
                else:
                    raise
            self.cdf_handles[file_date] = cdflib.CDF(file_path)
        #     self.data['epoch'] = np.concatenate((
        #         self.data['epoch'],
        #         np.array(cdflib.cdfepoch.to_datetime(_cdf['epoch']))
        #         )) 

        # idt = np.where(
        #     (self.data['epoch'] > self.time_range[0]) & (self.data['epoch'] <= self.time_range[1])
        #     )[0]
        # self.data['epoch'] = self.data['epoch'][idt]
        # for key in self.spectrum_keys:
        #     self.data[key] = self.data[key][idt, :]
        return self.cdf_handles

    def spectrum(self, component='MSCX', fce=True, ax=None, 
        pcolormesh_kwargs={}, spectrogram_kwargs={}):
        """
        Plots a spectrum of the EFW burst 1 waveform data.

        Parameters
        ----------
        component: str
            An electric field component. Can be "MSCX", "MSCY", or "MSCZ".
        fce: bool
            Plot the f_ce, f_ce/2, and f_ce/10 lines.
        ax: plt.Axes
            A subplot to plot on.
        pcolormesh_kwarg: dict
            The keyword arguments to pass into plt.pcolormesh. By default
            the only setting is a logarithmic color scale.
        spectrogram_kwargs: dict
            The keyword arguments to pass into scipy.signal.spectrogram.

        Returns
        -------
        matplotlib.collections.QuadMesh
            The pcolormesh object
        plt.Axes
            The subplot object.
        """
        if ax is None:
            _, ax = plt.subplots()

        if pcolormesh_kwargs is {}:
            pcolormesh_kwargs = {
                'norm':matplotlib.colors.LogNorm()
            }

        for times, f, psd in self.calc_spectrum(component, spectrogram_kwargs):
            p = ax.pcolormesh(times, f, 
                psd, shading='auto', **pcolormesh_kwargs)
        
        if not 'p' in locals():
            raise ValueError(f'No EFW burst 1 data from RBSP{self.sc_id} between {self.time_range}.')
        
        if fce:
            mag_data = Mag(self.sc_id, self.time_range)
            mag_data.load()
            _fce = mag_data.fce()
            for scaling, ls in zip([1, 0.5, 0.1], ['-', '--', ':']):
                ax.plot(mag_data['Epoch'][::100], scaling*_fce[::100], c='w', ls=ls)
        return p, ax

    def calc_spectrum(self, component, spectrogram_kwargs):
        """
        Generator to calculate the power spectral density of component.

        Parameters
        ----------
        component: str
            An electric field component. Must be "MSCX", "MSCY", or "MSCZ".
        spectrogram_kwargs: dict
            The keyword arguments to pass into scipy.signal.spectrogram.

        Yields
        ------
        np.array
            The timestamps in np.datetime64 format.
        np.array
            The spectrum frequencies.
        np.array
            The Phase space density.
        """
        idx = {"MSCX":0, "MSCY":1, "MSCZ":2}[component]
        for times, E in self.iter_chunks():
            frequency = 1/(pd.Timestamp(times[1]) - pd.Timestamp(times[0])).total_seconds()
            self.frequency, spec_times, psd = scipy.signal.spectrogram(E[:, idx], fs=frequency, **spectrogram_kwargs)
            yield spec_times, self.frequency, psd

    def _find_file(self, file_date):
        _file_match = (f'rbsp{self.sc_id.lower()}_l1_mscb1_{file_date:%Y%m%d}_v*.cdf')
        local_files = list(fbrbsp.config["rbsp_data_dir"].rglob(_file_match))

        if len(local_files) == 1:
            self.file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            url = (base_data_url + 
                f'rbsp{self.sc_id.lower()}/l1/efw/mscb1/{file_date.year}/'
                )
            downloader = sampex.Downloader(
                url,
                download_dir=fbrbsp.config["rbsp_data_dir"] / f'rbsp_{self.sc_id}' / \
                    'efw' / 'mscb1'
                )
            matched_downloaders = downloader.ls(match=_file_match)
            self.file_path = matched_downloaders[0].download(stream=True) 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} RBSP-{self.sc_id.upper()} EFW '
                f'mscb1 (burst 1) files found locally and online '
                f'that match {_file_match}.'
                )
        return self.file_path

    
    def iter_chunks(self, chunksize=1_000_000):
        """
        Iterate chunks of epochs and mscb1 variables.
        """
        first_cdf_handle = self.cdf_handles[list(self.cdf_handles.keys())[0]]
        n = len(self.cdf_handles.keys())*first_cdf_handle['Epoch'].shape[0]
        
        for _cdf in self.cdf_handles.values():
            i=0
            while i < n:
                before_start = cdflib.cdfepoch.to_datetime(
                    _cdf['epoch'][i+chunksize], to_np=True) < self.time_range[0]
                after_end = cdflib.cdfepoch.to_datetime(
                    _cdf['epoch'][i], to_np=True) > self.time_range[1]
                if before_start:
                    i+=chunksize
                    continue
                elif after_end:
                    break
                _epochs = _cdf.varget('epoch', startrec=i, endrec=i+chunksize)
                times = cdflib.cdfepoch.to_datetime(_epochs, to_np=True)
                yield times, _cdf.varget('mscb1', startrec=i, endrec=i+chunksize) # _cdf['mscb1'][i:i+chunksize, :]
                i+=chunksize

    # def __getitem__(self, _slice):
    #     """
    #     Access the cdf variables using keys (i.e., ['key']).
    #     """
    #     if isinstance(_slice, str):
    #         if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
    #             return self.data['epoch']
    #         elif _slice.lower() in [key.lower() for key in self.spectrum_keys]:
    #             return self.data[_slice]
    #         else:
    #             raise IndexError(f'{_slice} is not in the EFW Burst1 data.')
    #     else:
    #         raise IndexError(f'Only slicing with integer keys is supported.')

if __name__ == '__main__':
    efw = Burst1('B', ('2016-02-03T01', '2016-02-03T02'))
    efw.load()
    # for i, (times, E) in enumerate(efw.iter_chunks()):
    #     print(i,times, E, '\n\n\n')
    # print(efw['epoch'])
    # print(efw['BwSamples'])
    p, ax = efw.spectrum(
        spectrogram_kwargs={'nperseg':1024},
        # pcolormesh_kwargs={'norm':matplotlib.colors.LogNorm(vmin=1E-10, vmax=1E-3)}
        )
    plt.colorbar(p)
    # # ax.set_xlim(
    # #     dateutil.parser.parse('2016-01-20T19:41:06'),
    # #     dateutil.parser.parse('2016-01-20T19:41:14')
    # #     )
    # ax.set_ylim(1E2, 1E4)
    plt.show()
    pass