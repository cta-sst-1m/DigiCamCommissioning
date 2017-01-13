from ctapipe import visualization
import numpy as np
from matplotlib import pyplot as plt
from utils.histogram import histogram
from matplotlib.widgets import Button


class pickable_visu(visualization.CameraDisplay):
    def __init__(self, pickable_datas, extra_plot, figure, slice_func, show_fit, axis_scale,config, *args, **kwargs):
        super(pickable_visu, self).__init__(*args, **kwargs)
        self.pickable_datas = pickable_datas
        self.extra_plot = extra_plot
        self.figure = figure
        self.slice_func = slice_func
        self.show_fit = show_fit
        self.axis_scale = axis_scale
        self.config = config

    def on_pixel_clicked(self, pix_id):
        self.extra_plot.cla()
        colors = ['k', 'r', 'b']
        for i, pickable_data in enumerate(self.pickable_datas):
            slice = self.slice_func(pickable_data.data[pix_id],pickable_data.bin_centers) if self.slice_func else [0,
                                                                                         pickable_data.bin_centers.shape[
                                                                                           0], 1]
            '''
            init_func = pickable_data.fit_function
            if i == 1:
                init_func = lambda p,x : pickable_data.fit_function(p,x,config=self.config)
                func = lambda p, x: init_func(p, x, self.pickable_datas[0].fit_result[pix_id])
                pickable_data.fit_function = func
            '''

            pickable_data.show(which_hist=(pix_id,), axis=self.extra_plot,
                               show_fit=self.show_fit[i], slice=slice,
                               scale=self.axis_scale, color=colors[i], setylim=i == 0, config = self.config)
            '''
            if i == 1:
                pickable_data.fit_function = init_func
            '''
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')


class pickable_visu_mpe(visualization.CameraDisplay):
    def __init__(self, pickable_datas, extra_plot, figure, slice_func, level, show_fit, *args, **kwargs):
        super(pickable_visu_mpe, self).__init__(*args, **kwargs)
        self.pickable_datas = pickable_datas
        self.extra_plot = extra_plot
        self.figure = figure
        self.slice_func = slice_func
        self.level = level
        self.show_fit = show_fit
        self.pix_id = 700

    def on_pixel_clicked(self, pix_id):
        legend_handles = []
        self.pix_id = pix_id
        self.extra_plot.cla()
        axprev = plt.axes([0.7, 0.8, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.8, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.on_next_clicked)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.on_prev_clicked)
        for i, pickable_data in enumerate(self.pickable_datas):
            col = 'k' if i == 0 else 'b'

            slice = self.slice_func(pickable_data.data[self.level, self.pix_id],pickable_data.bin_centers)
            pickable_data.show(which_hist=(self.level, self.pix_id,), axis=self.extra_plot, show_fit=self.show_fit,
                               slice=slice)
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')

    def on_next_clicked(self, event):
        self.level += 1
        self.extra_plot.cla()
        for i, pickable_data in enumerate(self.pickable_datas):
            slice = self.slice_func(pickable_data.data[self.level, self.pix_id])
            pickable_data.show(which_hist=(self.level, self.pix_id,), axis=self.extra_plot, show_fit=self.show_fit,
                               slice=slice)
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')

    def on_prev_clicked(self, event):
        self.level -= 1
        if self.level < 0:
            self.level = 0
            return
        print('level', self.level)
        self.extra_plot.cla()
        for i, pickable_data in enumerate(self.pickable_datas):
            slice = self.slice_func(pickable_data.data[self.level, self.pix_id])
            pickable_data.show(which_hist=(self.level, self.pix_id,), axis=self.extra_plot, show_fit=self.show_fit,
                               slice=slice)
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')


class pickable_visu_led_mu(visualization.CameraDisplay):
    def __init__(self, pickable_datas, extra_plot, figure, slice_func, level, show_fit, *args, **kwargs):
        super(pickable_visu_led_mu, self).__init__(*args, **kwargs)
        self.pickable_datas = pickable_datas
        self.extra_plot = extra_plot
        self.figure = figure
        self.slice_func = slice_func
        self.level = level
        self.show_fit = show_fit

    def on_pixel_clicked(self, pix_id):
        print('clicked',pix_id)
        legend_handles = []
        self.pix_id = pix_id
        self.extra_plot.cla()
        for i, pickable_data in enumerate(self.pickable_datas):
            mu = pickable_data.fit_result[:, pix_id, self.level, 0]
            mu_err = pickable_data.fit_result[:, pix_id, self.level, 1]
            print(mu.shape, mu_err.shape)
            self.extra_plot.errorbar(np.arange(50, 260, 10), mu, yerr=mu_err, fmt='ok')
            self.extra_plot.set_ylim(1.e-1, 30.)
            # self.extra_plot.set_yscale('log')
            self.extra_plot.set_ylabel('<N(p.e.)>@DAC=x')
            self.extra_plot.set_xlabel('AC LED DAC')
            self.extra_plot.xaxis.get_label().set_ha('right')
            self.extra_plot.xaxis.get_label().set_position((1, 0))
            self.extra_plot.yaxis.get_label().set_ha('right')
            self.extra_plot.yaxis.get_label().set_position((0, 1))
            try:
                self.figure.canvas.draw()
            except ValueError:
                print('some issue to plot')


# Some display
def display(hists, geom,slice_func,pix_init=700,norm='lin',config=None):

    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu(hists, ax[1], fig, slice_func, [True], norm,config, geom, title='', norm='lin',
                                 cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    peak = hists[0].fit_result[:, 2, 0]
    peak[np.isnan(peak)] = 2.
    peak[peak < 0.] = 2.
    peak[peak > 10.] = 8.

    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = peak
    # noinspection PyProtectedMember
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(pix_init)
    plt.show()


def display_var(hist, geom,title='Gain [ADC/p.e.]', index_var=1, limit_min=0., limit_max=10., bin_width=0.2):
    f, ax = plt.subplots(1, 2, figsize=(20, 7))
    plt.subplot(1, 2, 1)
    vis_gain = visualization.CameraDisplay(geom, title='', norm='lin', cmap='viridis')
    vis_gain.add_colorbar()
    vis_gain.colorbar.set_label(title)
    h = np.copy(hist.fit_result[:, index_var, 0])
    h_err = np.copy(hist.fit_result[:, index_var, 1])
    h[np.isnan(h_err)] = limit_min
    h[h < limit_min] = limit_min
    h[h > limit_max] = limit_max
    vis_gain.image = h
    # plt.subplot(1,2,2)
    hh, bin_tmp = np.histogram(h, bins=np.arange(limit_min - bin_width / 2, limit_max + 1.5 * bin_width, bin_width))
    hh_hist = histogram(data=hh.reshape(1, hh.shape[0]),
                        bin_centers=np.arange(limit_min, limit_max + bin_width, bin_width), xlabel=title,
                        ylabel='$\mathrm{N_{pixel}/%.2f}$' % bin_width, label='All pixels')
    hh_hist.show(which_hist=(0,), axis=ax[1], show_fit=False)
    plt.show()
