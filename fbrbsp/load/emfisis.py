import json
import dateutil.parser

import numpy as np
import pandas as pd
import sampex
import cdflib

import fbrbsp

base_data_url = 'https://spdf.gsfc.nasa.gov/pub/data/rbsp/'

class Emfisis_spec:
    """
    Load a day of EMFISIS L2 spectrum data from the computer or the internet.

    Parameters
    ----------
    sc_id: str
        The spacecraft id, can be either A or B, case insensitive.
    inst: str
        Select between the wfr or hfr instruments.
    load_date: datetime.datetime, pd.Timestamp
        The date to load the data.

    Example
    -------
    
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
        Searches for and loads the HiRes data into memory.
        """
        self._file_match = (f'rbsp-{self.sc_id.lower()}_{self.inst.lower()}-'
            f'spectral-matrix-diagonal-merged_emfisis-l2_{self.load_date:%Y%m%d}_v*.cdf')
        self.file_path = self._find_file()
        return cdflib.CDF(self.file_path)

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

if __name__ == '__main__':
    emfisis = Emfisis_spec('A', 'WFR', '2015-02-02').load()
    pass