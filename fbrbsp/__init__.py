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
    fb_data_dir = settings['Paths'].get('fb_data_dir', None)
    rbsp_data_dir = settings['Paths'].get('rbsp_data_dir', None)

if fb_data_dir is None:
    fb_data_dir = pathlib.Path.home() / 'firebird-data'
if rbsp_data_dir is None:
    rbsp_data_dir = pathlib.Path.home() / 'rbsp-data'

config = {'here': here, 'fb_data_dir': fb_data_dir, 'rbsp_data_dir':rbsp_data_dir}