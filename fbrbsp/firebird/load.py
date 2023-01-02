import json
import pathlib
import dateutil.parser

import numpy as np
import pandas as pd
import sampex

import fbrbsp

data_url = 'https://solar.physics.montana.edu/FIREBIRD_II/'

class Hires:
    """
    Load a day of FIREBIRD-II HiRes data from the computer,
    and download the data if the file is not found.

    Parameters
    ----------
    sc_id: int
        The spacecraft id, can be either 3 or 4.
    load_date: datetime.datetime, pd.Timestamp
        The date to load the data.

    Example
    -------
    hr = Hires(3, '2015-02-02').load()
    """
    def __init__(self, sc_id, load_date) -> None:
        self.sc_id = sc_id
        if isinstance(load_date, str):
            self.load_date = dateutil.parser.parse(load_date)
        else:
            self.load_date = load_date
        return

    def load(self):
        """
        Searches for and loads the HiRes data into memory.
        """
        self._file_match = f"FU{self.sc_id}_Hires_{self.load_date:%F}_L2.txt"
        self.file_path = self._find_file()
        self.data = readJSONheadedASCII(self.file_path)
        return self.data

    def _find_file(self):
        local_files = list(fbrbsp.config["fb_data_dir"].rglob(self._file_match))

        if len(local_files) == 1:
            self.file_path = local_files[0]
        elif len(local_files) == 0:
            # File not found locally. Check online.
            downloader = sampex.Downloader(
                f'https://solar.physics.montana.edu/FIREBIRD_II/Data/FU_{self.sc_id}/hires/',
                download_dir=fbrbsp.config["fb_data_dir"] / f'FU_{self.sc_id}' / 'hires'
                )
            matched_downloaders = downloader.ls(match=self._file_match)
            self.file_path = matched_downloaders[0].download() 
        else:
            raise FileNotFoundError(
                f'{len(local_files)} HILT files found locally and online that match {self._file_match}.'
                )
        return self.file_path


def readJSONheadedASCII(file_path):
    """
    My simple implementation of spacepy.datamodel.readJSONheadedASCII that
    is specific for FIREBIRD-II data. You may use this if you can't install 
    spacepy for whatever reason.
    """
    # Read in the JSON header.
    header_list = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_list.append(line[1:])
            else:
                raw_data_str = f.readlines()

    # Massage the header
    clean_header_str = ''.join(header_list).replace('\n', '')
    parsed_header = json.loads(clean_header_str)
    # Massage the data
    raw_data_str = [row.replace('\n', '') for row in raw_data_str]
    # Parse times
    times_str = [row.split()[0] for row in raw_data_str]
    # Parse the other columns
    data_converted = np.array([row.split()[1:] for row in raw_data_str]).astype(float)

    data = fb_dict()
    data['Time'] = pd.to_datetime(times_str)
    for key in parsed_header:
        key_header = parsed_header[key]
        data.attrs[key] = key_header  # Save the attribute data.

        if key == 'Time':  # We already added Time.
            continue
        # Header key that correspond to columns
        if isinstance(key_header, dict):
            if len(key_header['DIMENSION']) != 1:
                raise NotImplementedError(
                    "readJSONheadedASCII doesn't implement columns with more than "
                    f"1 multidimensional. Got {key_header['DIMENSION']}."
                    )
            start_column = key_header['START_COLUMN']-1
            end_column = key_header['START_COLUMN']-1+key_header['DIMENSION'][0]
            if key_header['DIMENSION'][0] == 1:
                data[key] = data_converted[:, start_column]
            else:
                data[key] = data_converted[:, start_column:end_column]
        else:
            # Header key that correspond to global attributes
            if key in ['CADENCE', 'CAMPAIGN']:
                data.attrs[key] = float(key_header)
            else:
                data.attrs[key] = key_header
    return data

class fb_dict(dict):
    """
    Expand Python's dict class to include an attr attribute dictionary.
    
    Code credit goes to Matt Anderson:
    https://stackoverflow.com/questions/2390827/how-to-properly-subclass-dict-and-override-getitem-setitem
    (blame him for problems)
    """
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.attrs = {}
        return


if __name__ == "__main__":
    hr = Hires(3, '2015-02-02').load()
    pass