"""
Plot an example of inverse dispersion microburst that is in fact due to
DAPPER saturation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

from fbrbsp.dispersion.fit_dispersion import Bayes_Fit

plot_window_s=1 
time = '2015-02-02T06:12:26.310000'
channels = np.arange(6)

fb_id = 3
catalog_version=5
fit_interval_s = 0.3

model = Bayes_Fit(fb_id, channels=channels, catalog_version=catalog_version, 
                fit_interval_s=fit_interval_s, plot_window_s=plot_window_s, 
                full_ylabels=True)
model.load(time)
model.get_dispersion()
model.fit_dispersion(energy_dist='exp')

ax = model.plot()
ax[-1].set_xlim(200, 1000)
ax[-1].set_ylim(-100, 100)
loc = matplotlib.ticker.MaxNLocator(7) # this locator puts ticks at regular intervals
ax[-1].yaxis.set_major_locator(loc)
plt.show()
# print(f'{model.t0_diff_ms=}')
# print(f'{model.center_energy=}')