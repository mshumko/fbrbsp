import warnings
import pathlib
import configparser

__version__ = '0.0.1'

# Load the configuration settings.
here = pathlib.Path(__file__).parents[1].resolve()
settings = configparser.ConfigParser()
settings.read(here / 'config.ini')

# "Paths" only exist if config.ini exists
if settings.has_section('Paths'):
    fb_data_dir = settings['Paths'].get('fb_data_dir', None)
    rbsp_data_dir = settings['Paths'].get('rbsp_data_dir', None)

else:
    fb_data_dir = pathlib.Path.home() / 'firebird-data'
    rbsp_data_dir = pathlib.Path.home() / 'rbsp-data'

config = {'here': here, 'fb_data_dir': fb_data_dir, 'rbsp_data_dir':rbsp_data_dir}