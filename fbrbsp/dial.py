import numpy as np
import matplotlib.pyplot as plt

class Dial:
    def __init__(self, ax, angular_bins, radial_bins, H):
        """ 
        This class makes a dial (polar) plot were MLT is the azimuthal
        coordinate and L shell is the radial coordinate. 

        Parameters
        ----------
        TODO: Write

        Example
        -------
        TODO: Write
        """
        self.ax = ax
        self.angular_bins = angular_bins
        self.radial_bins = radial_bins
        self.H = H

        if 'Polar' not in str(type(ax)):
            raise ValueError('Subplot is not polar. For example, '
                'create ax with \n ax[0] = plt.subplot(121, projection="polar")')
        return

    def draw_dial(self, colorbar=True, L_labels=[2,4,6,8], mesh_kwargs={}, colorbar_kwargs={}):
        """
        Draws a dial plot on the self.ax subplot object (must have projection='polar' kwarg). 
        colorbar=True - Plot the colorbar or not.
        L_labels=[2,4,6,8] - What L labels to plot
        mesh_kwargs={} - The dictionary of kwargs passed to plt.pcolormesh() 
        colorbar_kwargs={} - The dictionary of kwargs passed into plt.colorbar()
        """
        self.L_labels = L_labels
        # Turn off the grid to prevent a matplotlib deprecation warning 
        # (see https://matplotlib.org/3.5.1/api/prev_api_changes/api_changes_3.5.0.html#auto-removal-of-grids-by-pcolor-and-pcolormesh)
        self.ax.grid(False) 
        angular_grid, radial_grid = np.meshgrid(self.angular_bins, self.radial_bins)
        p = self.ax.pcolormesh(angular_grid*np.pi/12, radial_grid, self.H, **mesh_kwargs)
        

        self.draw_earth()
        self._plot_params()

        if colorbar:
            plt.colorbar(p, ax=self.ax, **colorbar_kwargs)
        return

    def draw_earth(self, earth_resolution=50):
        """ Given a subplot object, draws the Earth with a shadow"""
        # Just x,y coords for a line (to map to polar coords)
        earth_circ = (np.linspace(0, 2*np.pi, earth_resolution), np.ones(earth_resolution)) 
        # x, y_lower, y_upper coords for Earth's shadow (maps to polar).
        earth_shadow = (
                        np.linspace(-np.pi/2, np.pi/2, earth_resolution), 
                        0, 
                        np.ones(earth_resolution)
                        )
        self.ax.plot(*earth_circ, c='k')
        self.ax.fill_between(*earth_shadow, color='k')
        return

    def _plot_params(self):
        # Draw L shell contours and get L and MLT labels 
        L_labels_names = self._draw_L_contours()
        mlt_labels = np.round(self.ax.get_xticks()*12/np.pi).astype(int)
        self.ax.set_xlabel('MLT')
        self.ax.set_theta_zero_location("S") # Midnight at bottom
        self.ax.set_xticks(mlt_labels*np.pi/12, labels=mlt_labels)
        self.ax.set_yticks(self.L_labels)
        self.ax.set_yticklabels(L_labels_names, fontdict={'horizontalalignment':'right'})
        return

    def _draw_L_contours(self, earth_resolution=50):
        """ Plots a subset of the L shell contours. """
        # Draw azimuthal lines for a subset of L shells.
        L_labels_names = [str(i) for i in self.L_labels[:-1]] + [f'L = {self.L_labels[-1]}']
        # L_labels_names = [str(i) for i in self.L_labels]
        for L in self.L_labels:
            self.ax.plot(np.linspace(0, 2*np.pi, earth_resolution), 
                        L*np.ones(earth_resolution), ls=':', c='k')
        return L_labels_names