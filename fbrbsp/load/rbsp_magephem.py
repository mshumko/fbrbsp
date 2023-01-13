from collections import defaultdict

import numpy as np
import sampex
import cdflib

import fbrbsp
import fbrbsp.load.utils as utils

base_data_url = 'https://spdf.gsfc.nasa.gov/pub/data/rbsp/'

class MagEphem:
    """
    Loads and RBSP magnetic ephemeris cdf data provided by the ECT team at LANL.

    Parameters
    ----------
    sc_id: str
        The spacecraft id, can be either A or B, case insensitive.
    model: str
        The magnetic field model. Can be "op77q", "t89d", "t89q", or "ts04d".
    time_range: list[str, datetime.datetime, or pd.Timestamp]
        A list, array, or tuple defining the date and time bounds to load.

    Methods
    -------
    load()
        Searches for and loads the magnetic ephemeris.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> rbsp_id = 'A'
    >>> time_range = ('2016-01-02T00', '2016-01-5T15:30')
    >>> mag = MagEphem(rbsp_id, 't89d', time_range)
    >>> mag.load()
    >>> print(mag['epoch'])
    [datetime.datetime(2016, 1, 2, 0, 1) datetime.datetime(2016, 1, 2, 0, 2)
    datetime.datetime(2016, 1, 2, 0, 3) ...
    datetime.datetime(2016, 1, 5, 15, 28)
    datetime.datetime(2016, 1, 5, 15, 29)
    datetime.datetime(2016, 1, 5, 15, 30)]
    >>> print(mag['L_Label'])
    [['L_90' 'L_85' 'L_80' 'L_75' 'L_70' 'L_65' 'L_60' 'L_55' 'L_50' 'L_45'
    'L_40' 'L_35' 'L_30' 'L_25' 'L_20' 'L_15' 'L_10' 'L_5 ']]
    >>> print(mag['L'])
    [[ 1.12443e+00  1.12429e+00  1.12387e+00 ... -1.00000e+31 -1.00000e+31
    -1.00000e+31]
    [ 1.14276e+00  1.14264e+00  1.14230e+00 ... -1.00000e+31 -1.00000e+31
    -1.00000e+31]
    [ 1.16401e+00  1.16392e+00  1.16365e+00 ... -1.00000e+31 -1.00000e+31
    -1.00000e+31]
    ...
    [ 4.55085e+00  4.55027e+00  4.54856e+00 ...  4.48651e+00  4.48802e+00
    -1.00000e+31]
    [ 4.53175e+00  4.53117e+00  4.52945e+00 ...  4.46747e+00  4.46905e+00
    -1.00000e+31]
    [ 4.51248e+00  4.51190e+00  4.51017e+00 ...  4.44829e+00  4.44993e+00
    -1.00000e+31]]
    >>> 
    >>> # Plot the L-shell for equatorial-mirroring particles.
    >>> pa_idx = 0
    >>> pa = mag["L_Label"][0, pa_idx].split('_')[-1]
    >>> plt.plot(mag['epoch'], mag['L'][:, pa_idx])
    >>> plt.title(rf'RBSP-{rbsp_id} L($\alpha$={pa}) vs time')
    >>> plt.show()
    """
    def __init__(self, sc_id, model, time_range) -> None:
        self.sc_id = sc_id.lower()
        self.model = model.lower()
        self.time_range = time_range
        assert self.model in ["op77q", "t89d", "t89q", "ts04d"]
        self.time_range = utils.validate_time_range(time_range)
        return

    def load(self, missing_ok=True):
        """
        Searches for and loads the data into memory.
        """
        file_dates = utils.get_filename_times(self.time_range, dt='days')
        self.cdf_handles = {}

        for file_date in file_dates:
            try:
                file_path = self._find_file(file_date)
            except FileNotFoundError as err:
                if missing_ok and ('magnetic ephemeris files found locally and online' in str(err)):
                    continue
                else:
                    raise
            _cdf = cdflib.CDF(file_path)
            self.cdf_handles[file_date] = _cdf
        return self.cdf_handles

    def _find_file(self, file_date):
        _file_match = (f'rbsp-{self.sc_id.lower()}_mag-ephem_def-1min-'
            f'{self.model}_{file_date:%Y%m%d}_v*.cdf')
        local_files = list(fbrbsp.config["rbsp_data_dir"].rglob(_file_match))

        if len(local_files) == 1:
            self.file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            url = (base_data_url + 
                f'rbsp{self.sc_id.lower()}/ephemeris/ect-mag-ephem/cdf/' +
                f'def-1min-{self.model}/{file_date.year}/'
                )
            downloader = sampex.Downloader(
                url,
                download_dir=fbrbsp.config["rbsp_data_dir"] / f'rbsp_{self.sc_id}' / \
                    'magephem' / self.model
                )
            matched_downloaders = downloader.ls(match=_file_match)
            self.file_path = matched_downloaders[0].download() 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} RBSP-{self.sc_id.upper()} magnetic ephemeris files '
                f'found locally and online that match {_file_match}.'
                )
        return self.file_path

    def __getitem__(self, _slice):
        """
        Access the cdf variables using keys (i.e., ['key']).
        """
        first_cdf_handle = self.cdf_handles[list(self.cdf_handles.keys())[0]]
        if isinstance(_slice, str):
            if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
                if hasattr(self, 'epoch'):
                    return self.epoch  # Don't recompute epochs.
                else:
                    self._get_epochs()
                return self.epoch
            elif _slice in first_cdf_handle.cdf_info()['zVariables']:
                return self._get_variable(_slice)
            else:
                raise IndexError(f'{_slice} variable is not in the '
                    f'magnetic ephemeris {self.model} data. Available variables: '
                    f"{first_cdf_handle.cdf_info()['zVariables']}."
                    )
        else:
            raise IndexError(f'Only slicing with integer keys is supported.')

    def _get_epochs(self):
        self.epoch = np.array([])
        for _cdf in self.cdf_handles.values():
            self.epoch = np.concatenate(
                (self.epoch, np.array(cdflib.cdfepoch.to_datetime(_cdf['epoch'])))
                )
        self.idt = np.where(
            (self.epoch > self.time_range[0]) & (self.epoch <= self.time_range[1])
            )[0]
        self.epoch = self.epoch[self.idt]
        return

    def _get_variable(self, _slice):
        first_cdf_handle = self.cdf_handles[list(self.cdf_handles.keys())[0]]
        if 'label' in _slice.lower():
            # No appending necessary since there are just labels describing variables.
            return first_cdf_handle[_slice]
        else:
            for _cdf in self.cdf_handles.values():
                if '_data' not in locals():
                    _data = _cdf[_slice]
                else:
                    _data = np.concatenate((_data, _cdf[_slice]))
        # Filter by time
        if not hasattr(self, 'idt'):
            self._get_epochs()
        return _data.take(indices=self.idt, axis=0)