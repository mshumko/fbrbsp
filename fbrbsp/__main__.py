import sys
import pathlib
import configparser

# Run the configuration script with
# python3 -m fbrbsp [init, initialize, config, or configure]

here = pathlib.Path(__file__).parent.resolve()


def check_directory(dir_str, default_dir):
    """
    Checks if the user-specified dir_str is empty or not. If empty, it will 
    check the default directory specified by ~/default_dir/.
    """
    if dir_str == '':
        dir_path = pathlib.Path(default_dir)
    else:
        dir_path = pathlib.Path(dir_str)
    
    if not pathlib.Path(dir_path).exists():
        pathlib.Path(dir_path).mkdir(parents=True)
        print(f'Made {dir_path} directory.')
    else:
        print(f'The {dir_path} directory already exists.')
    return dir_path


if (len(sys.argv) > 1) and (sys.argv[1] in ['init', 'initialize', 'config', 'configure']):
    print('Running the fbrbsp configuration script.')
    
    s = (
        f"Where is the FIREBIRD-II data directory? Press enter for the default "
        f"directory ~/firebird-data/.\n"
    )
    fb_data_dir = input(s)
    fb_data_dir = check_directory(fb_data_dir, pathlib.Path.home() / 'firebird-data')

    s = (
        f"Where is the RBSP data directory? Press enter for the default "
        f"directory ~/rbsp-data/.\n"
    )
    rbsp_data_dir = input(s)
    rbsp_data_dir = check_directory(rbsp_data_dir, pathlib.Path.home() / 'rbsp-data')

    # Create a configparser object and add the user configuration.
    config = configparser.ConfigParser()
    config['Paths'] = {
        'here':here,
        'fb_data_dir': fb_data_dir,
        'rbsp_data_dir': rbsp_data_dir
        }

    with open(here / 'config.ini', 'w') as f:
        config.write(f)

else:
    print(
        'This is a configuration script to set up the fbrbsp package. To '
        'configure this package run "python3 -m project config".'
    )