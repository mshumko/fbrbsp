import pathlib
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

import fbrbsp
from fbrbsp.dial import Dial


L = np.array([])
MLT = np.array([])
minMLT = np.array([])

for fb_id in [3, 4]:
    for rbsp_id in ['A', 'B']:
        file_name = f'FU{fb_id}_RBSP{rbsp_id.upper()}_conjunctions_dL10_dMLT10_final_hr.csv'
        catalog_path = fbrbsp.config['here'].parent / 'data' / file_name
        catalog = pd.read_csv(catalog_path)

        # meanL,meanMLT,minMLT
        MLT = np.append(MLT, catalog['meanMLT'].to_numpy())
        L = np.append(L, catalog['meanL'].to_numpy())
        minMLT = np.append(minMLT, catalog['minMLT'].to_numpy())

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5,5))
_dial = Dial(ax, None, None, None)
_dial.L_labels = [2,4,6,8]
ax.grid(False) 
_dial.draw_earth()
_dial._plot_params()
s = ax.scatter(MLT, L, c=minMLT)
plt.colorbar(s, label=f'Minimum ${{\Delta}}$MLT')
ax.set_title('FIREBIRD-II/RBSP Conjunctions')
plt.tight_layout()
plt.show()