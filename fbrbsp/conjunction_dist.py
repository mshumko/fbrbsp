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

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
_dial = Dial(ax, None, None, None)
_dial.L_labels = [2,4,6,8]
ax.grid(False) 
_dial.draw_earth()
_dial._plot_params()

plt.show()