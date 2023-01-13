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
        if isinstance(_slice, str):
            if ("epoch" in _slice.lower()) or ("time" in _slice.lower()):
                if hasattr(self, 'epoch'):
                    return self.epoch  # Don't recompute epochs.
                self.epoch = np.array([])
                for _cdf in self.cdf_handles.values():
                    self.epoch = np.concatenate(
                        (self.epoch, np.array(cdflib.cdfepoch.to_datetime(_cdf['epoch'])))
                        )
                self.idt = np.where(
                    (self.epoch > self.time_range[0]) & (self.epoch <= self.time_range[1])
                    )[0]
                self.epoch = self.epoch[self.idt]
                return self.epoch
            elif _slice.lower() in self.data.keys():
                # return self.data[_slice]
                pass
            else:
                raise IndexError(f'{_slice} variable is not in the '
                    f'magnetic ephemeris {self.model} data.'
                    )
        else:
            raise IndexError(f'Only slicing with integer keys is supported.')

if __name__ == '__main__':
    mag = MagEphem('A', 't89d', ('2016-01-02T00', '2016-01-03T15:30'))
    mag.load()
    print(mag['epoch'])
    # print(mag[''])