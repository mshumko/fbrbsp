"""
Plot a grid of dials showing the amount of EMFISIS and EFW burst data available 
during each conjunction. 
"""
import pathlib
import string

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import fbrbsp
from fbrbsp.dial import Dial

RB='RBSPA'
FU='FU3'

file_dir = pathlib.Path(fbrbsp.__file__).parents[1] / 'data'
file_name = f'{RB}_{FU}_conjunction_values_hr.csv'    #Preeti's version
file_path = file_dir / file_name
assert file_path.exists()
data = pd.read_csv(file_dir / file_name, delimiter=',',skiprows=0)

ntimes = len(data['Stop_Date_time'])
#Convert to datetime
dt_closest = [datetime.strptime(data['Closest_Date_time'][i], "%Y-%m-%dT%H:%M:%S") for i in range(ntimes)]
dt_start = [datetime.strptime(data['Start_Date_time'][i], "%Y-%m-%dT%H:%M:%S") for i in range(ntimes)]
dt_stop = [datetime.strptime(data['Stop_Date_time'][i], "%Y-%m-%dT%H:%M:%S") for i in range(ntimes)]

# #Burst data and conjunctions
# plt.plot(dt_closest,data['EMFb'],'*',dt_closest,data['B1b'],'*',dt_closest,data['B2b'],'*')
# plt.yscale('log')
# plt.ylabel('seconds of burst data\nBlue=EMFISIS\nOrange=EFW B1; Green=EFW B2')
# plt.title(RB+' - '+FU + ' conjunctions')

# #Chorus lowerband amplitude and conjunctions
# plt.plot(dt_closest,data['SpecBMax_lb'],'*')
# plt.yscale('log')
# plt.ylabel('Lower band chorus amp (nT^2/Hz)\nfrom +/- 1 hr of conjunction')
# plt.title(RB+' - '+FU + ' conjunctions')

#-------------------------------------------------------------------
#Make scatter plots of interesting values for Mike Shumko
#-------------------------------------------------------------------

b1b = data['B1b']
b2b = data['B2b']
b1b[b1b == 0] = np.nan
b2b[b2b == 0] = np.nan
logv = np.log(data['SpecBMax_lb'])
logb1sec = np.log(data['B1b'])
logb2sec = np.log(data['B2b'])

fig=plt.figure(figsize=(8,8))
ax = np.zeros((2,2), dtype=object)

ax[0, 0]=fig.add_subplot(221,projection='polar')
ax[0, 1]=fig.add_subplot(222,projection='polar')
ax[1, 0]=fig.add_subplot(223,projection='polar')
ax[1, 1]=fig.add_subplot(224,projection='polar')

for ax_i in ax.flatten():
    _dial = Dial(ax_i, None, None, None)
    _dial.L_labels = [2,4,6,8]
    ax_i.grid(False) 
    _dial.draw_earth()
    _dial._plot_params()

fig.suptitle(f'Wave Burst Availability during {RB}-{FU} conjunctions')
scatter_objects = np.zeros_like(ax)
scatter_objects[0, 0] = ax[0, 0].scatter(data['MLTrb'],data['Lrb'],c=data['EMFb'],vmin=0,vmax=np.nanmax(data['EMFb'])/2)
scatter_objects[0, 1] = ax[0, 1].scatter(data['MLTrb'],data['Lrb'],c=b1b,vmin=np.nanmin(b1b),vmax=np.nanmax(b1b)/1.5)
scatter_objects[1, 0] = ax[1, 0].scatter(data['MLTrb'],data['Lrb'],c=b2b,vmin=np.nanmin(b2b),vmax=np.nanmax(b2b)/2)
scatter_objects[1, 1] = ax[1, 1].scatter(data['MLTrb'],data['Lrb'],c=logv,vmin=np.min(logv),vmax=np.max(logv))

titles = [
    'EMFISIS burst data',
    'EFW burst 1 data',
    'EFW burst 2 data',
    'Peak Amplitude of lower-band chorus'
]
colorbar_labels = [
    '[sec]',
    '[sec]',
    '[sec]',
    '[log10(pT^2/Hz)]'
]
_z = zip(ax.flatten(), scatter_objects.flatten(), string.ascii_uppercase, titles, colorbar_labels)
for ax_i, scatter_object, title_label, title, label in _z:
    plt.colorbar(scatter_object, ax=ax_i, label=label)
    ax_i.set_title(f'({title_label}) {title}')

plt.tight_layout()
plt.show()

