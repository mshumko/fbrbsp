import warnings
import pathlib
import configparser

__version__ = '0.0.1'

# Load the configuration settings.
here = pathlib.Path(__file__).parent.resolve()
settings = configparser.ConfigParser()
settings.read(here / 'config.ini')

# Go here if config.ini exists (don't crash if the project is not yet configured.)
if 'Paths' in settings:  
    try:
        fb_data_dir = settings['Paths']['fb_data_dir']
        rbsp_data_dir = settings['Paths']['fb_data_dir']
    except KeyError as err:
        warnings.warn('The firebird package did not find the config.ini file. '
            'Did you run "python3 -m firebird config"?')

    config = {'here': here, 'fb_data_dir': fb_data_dir}