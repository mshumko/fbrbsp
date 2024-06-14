"""
Load the list of chorus elements identified with Jaibei's code.
"""
import pathlib

import h5py
import scipy.io

import fbrbsp

file_dir = pathlib.Path(fbrbsp.__file__).parents[1] / 'data' / 'Conjunction_chorus'
file_name = f'Mike_chorus3.mat'
file_path = file_dir / file_name
assert file_path.exists()
data = h5py.File(file_path, 'r')
data2 = scipy.io.loadmat(file_path)
pass