import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from numpy.linalg import inv

class Histogram:
    """
    A simple class to hold histograms data, manipulate and fit them

    This class has been designed to treat many histograms having the same x-bins at the same time

    """

    def __init__(self, data=np.zeros(0), data_shape=(0,), bin_centers=np.zeros(0), bin_center_min=0, bin_center_max=1,
                 bin_width=1, xlabel='x', ylabel='y', label='hist'):
        """
        Initialise method

        Histogram can be either initialised from existing histograms:
          - data
          - bin_centers
        or from the data shape and the binning:
          - data_shape
          - (bin_width, bin_center_min, bin_center_max) or bin_centers numpy array

        :param data: the numpy array holding the bin yields
        :param data_shape: the shape of the Histogram table (ie. data will be of shape data_shape+bin_centers.shape)
        :param bin_centers: the numpy array of the bin centers
        :param bin_center_min: the minimal bin centers
        :param bin_center_max: the maximal bin centers
        :param bin_width: the bin width
        :param xlabel: the x axis label
        :param ylabel: the y axis label
        :param label: the base label for the histograms
        """
        # Initialisation of the bin_centers
        if bin_centers.shape[0] == 0:
            self.bin_width = bin_width
            # generate the bin edge array
            self.bin_edges = np.arange(bin_center_min - self.bin_width / 2, bin_center_max + self.bin_width / 2 + 1,
                                       self.bin_width)
            # generate the bin center array
            self.bin_centers = np.arange(bin_center_min, bin_center_max + 1, self.bin_width)
        else:
            self.bin_centers = bin_centers
            # generate the bin edge array
            self.bin_width = self.bin_centers[1] - self.bin_centers[0]
            self.bin_edges = np.arange(self.bin_centers[0] - self.bin_width / 2,
                                       self.bin_centers[self.bin_centers.shape[0] - 1] + self.bin_width / 2 + 1,
                                       self.bin_width)

        # Initialisation of data
        if data.shape[0] == 0:
            self.data = np.zeros(data_shape + (self.bin_centers.shape[0],))
            self.errors = np.zeros(data_shape + (self.bin_centers.shape[0],))
        else:
            self.data = data
            self._compute_errors()

        # Initialisation of fit results and labels
        self.fit_result = None
        self.fit_chi2_ndof = None
        self.fit_function = None
        self.fit_axis = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label

    def fill(self, value, indices=None):
        """
        Update the Histogram array with an array of values
        :param value:
        :param indices:
        :return: void
        """
        # change the value array to an array of Histogram index to be modified
        hist_indices = ((value - self.bin_edges[0]) // self.bin_width).astype(int)
        # treat overflow and underflow
        hist_indices[hist_indices > self.data.shape[-1] - 1] = self.data.shape[-1] - 1
        hist_indices[hist_indices < 0] = 0
        # get the corresponding indices multiplet
        dim_indices = tuple([np.indices(value.shape)[i].reshape(np.prod(value.shape)) for i in
                             range(np.indices(value.shape).shape[0])], )
        dim_indices += (hist_indices.reshape(np.prod(value.shape)),)
        if value[..., 0].shape == self.data[..., 0].shape or not indices:
            self.data[dim_indices] += 1
        else:
            self.data[indices][dim_indices] += 1

    def fill_with_batch(self, batch, indices=None):
        """
        A function to transform a batch of data in Histogram and add it to the existing one
        :param batch: a np.array with the n-1 same shape of data, and n dimension containing the array to Histogram
        :param indices: a tuple limiting the filled data to data[indices]
        :return:
        """
        # noinspection PyUnusedLocal
        data = None
        if not indices:
            data = self.data
        else:
            data = self.data[indices]

        hist = lambda x: np.histogram(x, bins=self.bin_edges, density=False)[0]

        if batch.dtype != 'object':
            # Get the new Histogram
            new_hist = np.apply_along_axis(hist, len(data.shape) - 1, batch)
            # Add it to the existing
            data = np.add(data, new_hist)

        else:
            for index in np.ndindex(batch.shape):
                # Get the new Histogram
                new_hist = hist(batch[index])
                # Add it to the existing
                data[index] = np.add(data[index], new_hist)

        if not indices:
            self.data = data
        else:
            self.data[indices] = data

        self._compute_errors()

    @staticmethod
    def _residual(function, p, x, y, y_err):
        """
        Return the residuals of the data with respect to a function

        :param function: The function (defined with arguments params and x)
        :param p: the parameters of the function (np.array)
        :param x: the x values (np.array)
        :param y: the y values (np.array)
        :param y_err: the y values errors (np.array)
        :return: the residuals (np.array)
        """
        return (y - function(p, x)) / y_err

    def _compute_errors(self):
        self.errors = np.sqrt(self.data)
        self.errors[self.errors == 0.] = 1.

    def _axis_fit(self, idx, func, p0, slice_list=None, bounds=None, fixed_param=None, verbose=False):
        """
        Perform a fit on this specific Histogram

        :param idx:      the index of the Histogram in self.data        (type: tuple)
        :param func:     the fit function                               (type: function)
        :param p0:       the initial fit parameters                     (type: list)
        :param slice_list:    the slice_list of data to fit                       (type list)
        :param bounds:   the boundary for the parameters                (type tuple(list,list))
        :param fixed_param: the parameters to be fixed and their values (type list(list,list))
        :return: the fit result                                         (type np.array)
        """
        # Reduce the functions parameters according to the fixed_param
        reduced_p0 = p0
        reduced_bounds = bounds
        reduced_func = func
        # TODO optimize this part
        if type(fixed_param).__name__ != 'NoneType':
            def reduced_func(p, x, *args, **kwargs):
                p_new, j = [], 0
                for param_i, param_val in enumerate(p0):
                    if not (param_i in fixed_param[0]):
                        p_new += [p[j]]
                        j += 1
                    else:
                        for value_i, value in enumerate(fixed_param[0]):
                            if value == param_i:
                                p_new += [fixed_param[1][value_i]]
                return func(p_new, x, *args, **kwargs)

            reduced_p0 = []
            for i, param in enumerate(p0):
                if not (i in fixed_param[0]):
                    reduced_p0 += [param]
            # print('red_p0',reduced_p0)

            reduced_bounds = [[], []]
            for i, param in enumerate(p0):
                if not (i in fixed_param[0]):
                    reduced_bounds[0] += [bounds[0][i]]
                    reduced_bounds[1] += [bounds[1][i]]
            reduced_bounds = tuple(reduced_bounds)
            # print('red_bound',reduced_bounds)
        # noinspection PyUnusedLocal
        fit_result = None
        if slice_list == [0, 0, 1] or self.data[idx][slice_list[0]:slice_list[1]:slice_list[2]].shape == 0 \
                or np.any(np.isnan(reduced_p0)) \
                or np.any(np.isnan(reduced_bounds[0])) or np.any(np.isnan(reduced_bounds[1])) \
                or np.any(np.isnan(p0)):
            if verbose:
                print('Bad inputs')
            fit_result = (np.ones((len(reduced_p0), 2)) * np.nan)
            ndof = (slice_list[1] - slice_list[0]) / slice_list[2] - len(reduced_p0)
            chi2 = np.nan
        else:
            if not slice_list:
                slice_list = [0, self.bin_centers.shape[0] - 1, 1]
            ndof = (slice_list[1] - slice_list[0]) / slice_list[2] - len(reduced_p0)
            chi2 = np.nan
            try:
                residual = lambda p, x, y, y_err: self._residual(reduced_func, p, x, y, y_err)
                out = scipy.optimize.least_squares(residual, reduced_p0, args=(
                    self.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]],
                    self.data[idx][slice_list[0]:slice_list[1]:slice_list[2]],
                    self.errors[idx][slice_list[0]:slice_list[1]:slice_list[2]]), bounds=reduced_bounds)
                # noinspection PyUnresolvedReferences
                val = out.x
                # noinspection PyUnresolvedReferences,PyUnresolvedReferences
                chi2 = np.sum(out.fun * out.fun)
                try:
                    # noinspection PyUnresolvedReferences,PyUnresolvedReferences
                    cov = np.sqrt(np.diag(inv(np.dot(out.jac.T, out.jac))))
                    fit_result = np.append(val.reshape(val.shape + (1,)), cov.reshape(cov.shape + (1,)), axis=1)
                except np.linalg.linalg.LinAlgError as inst:
                    if verbose:
                        print(inst)
                    print('Could not compute errors')
                    print(inst)
                    fit_result = np.append(val.reshape(val.shape + (1,)), np.ones((len(reduced_p0), 1)) * np.nan,
                                           axis=1)

            except Exception as inst:
                print('failed fit', inst, 'index', idx)
                print('p0', reduced_p0)
                print('bound min', reduced_bounds[0])
                print('bound max', reduced_bounds[1])
                fit_result = (np.ones((len(reduced_p0), 2)) * np.nan)

        # restore the fixed_params in the fit_result
        if type(fixed_param).__name__ != 'NoneType':
            for k, i in enumerate(fixed_param[0]):
                fit_result = np.insert(fit_result, int(i), [fixed_param[1][k], 0.], axis=0)
        return fit_result, chi2, ndof

    # noinspection PyDefaultArgument
    def fit(self, func, p0_func, slice_func, bound_func, config=None, limited_indices=None, fixed_param=[],
            force_quiet=False):
        """
        An helper to fit Histogram
        :param p0_func:
        :param slice_func:
        :param bound_func:
        :param config:
        :param limited_indices:
        :param fixed_param:
        :param force_quiet:
        :param func:
        :return:
        """
        data_shape = list(self.data.shape)
        data_shape.pop()
        data_shape = tuple(data_shape)
        self.fit_function = func
        # self.fit_result = None
        # perform the fit of the 1D array in the last dimension
        count = 0
        indices_list = np.ndindex(data_shape)
        if limited_indices:
            indices_list = limited_indices
        for indices in indices_list:
            if type(self.fit_result).__name__ != 'ndarray':
                if type(config).__name__ != 'ndarray':
                    self.fit_result = np.ones(
                        data_shape + (len(p0_func(self.data[indices], self.bin_centers, config=None)), 2)) * np.nan
                else:
                    self.fit_result = np.ones(data_shape + (len(p0_func(self.data[indices], self.bin_centers,
                                                                        config=config[indices])), 2)) * np.nan

            if type(self.fit_chi2_ndof).__name__ != 'ndarray':
                self.fit_chi2_ndof = np.ones(data_shape + (2,)) * np.nan

            if not force_quiet:
                print("Fit Progress {:2.1%}".format(count / np.prod(data_shape)), end="\r")
            count += 1
            # noinspection PyUnusedLocal
            fit_res = None
            # treat the fixed parameters
            list_fixed_param = None
            if len(fixed_param) > 0:
                list_fixed_param = [[], []]
                for p in fixed_param:
                    list_fixed_param[0].append(p[0])
                    if isinstance(p[1], tuple):
                        list_fixed_param[1].append(config[indices][p[1]])
                    else:
                        list_fixed_param[1].append(p[1])

            if type(config).__name__ != 'ndarray':
                fit_res, chi2, ndof = self._axis_fit(indices, func,
                                                     p0_func(self.data[indices], self.bin_centers, config=None),
                                                     slice_list=slice_func(self.data[indices], self.bin_centers,
                                                                           config=None),
                                                     bounds=bound_func(self.data[indices], self.bin_centers,
                                                                       config=None),
                                                     fixed_param=list_fixed_param)

            else:
                func_reduced = lambda _p, x: func(_p, x, config=config[indices])
                # print('slice',slice_func(self.data[indices], self.bin_centers, config=config[indices]))
                # print('bounds',bound_func(self.data[indices], self.bin_centers,config=config[indices]))
                # print('fixed_param',list_fixed_param)

                fit_res, chi2, ndof = self._axis_fit(indices, func_reduced,
                                                     p0_func(self.data[indices], self.bin_centers,
                                                             config=config[indices]),
                                                     slice_list=slice_func(self.data[indices], self.bin_centers,
                                                                           config=config[indices]),
                                                     bounds=bound_func(self.data[indices], self.bin_centers,
                                                                       config=config[indices]),
                                                     fixed_param=list_fixed_param)
            # make sure sizes matches
            if self.fit_result[indices].shape[-2] < fit_res.shape[-2]:
                num_column_to_add = fit_res.shape[-2] - self.fit_result[indices].shape[-2]
                additional_shape = list(self.fit_result.shape)
                additional_shape[-2] = num_column_to_add
                additional_columns = np.zeros(tuple(additional_shape), dtype=fit_res.dtype)
                self.fit_result = np.append(self.fit_result, additional_columns, axis=-2)

            if self.fit_result[indices].shape[-2] > fit_res.shape[-2]:
                num_column_to_add = self.fit_result[indices].shape[-2] - fit_res.shape[-2]
                additional_shape = list(fit_res.shape)
                additional_shape[-2] = num_column_to_add
                additional_columns = np.zeros(tuple(additional_shape), dtype=self.fit_result.dtype)
                fit_res = np.append(fit_res, additional_columns, axis=-2)

            self.fit_result[indices] = fit_res
            self.fit_chi2_ndof[indices][0] = chi2
            self.fit_chi2_ndof[indices][1] = ndof

    def find_bin(self, x):
        return (x - self.bin_edges[0]) // self.bin_width

    def show(self, which_hist=None, axis=None, show_fit=False, slice_list=None, config=None, scale='linear', color='k',
             set_ylim=True):

        if not which_hist:
            which_hist = (0,) * len(self.data[..., 0].shape)
        if not slice_list:
            slice_list = [0, self.bin_centers.shape[0], 1]
        if scale == 'log':
            x_text = np.min(self.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]]) + 0.5
            y_text = 0.1 * (np.min(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]) + self.errors[
                which_hist + (np.argmax(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]),)])

        else:
            x_text = np.min(self.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]])
            y_text = 0.8 * (np.max(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]) + self.errors[
                which_hist + (np.argmax(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]),)])

        text_fit_result = ''
        precision = int(3)

        if show_fit:
            for i in range(self.fit_result[which_hist].shape[-2]):
                text_fit_result += 'p' + str(i) + ' : ' + str(
                    np.round(self.fit_result[which_hist + (i, 0,)], precision))
                text_fit_result += ' $\pm$ ' + str(np.round(self.fit_result[which_hist + (i, 1,)], precision))
                text_fit_result += '\n'

            text_fit_result += '$\chi^2$ = ' + str(np.round(self.fit_chi2_ndof[which_hist][0], precision)) + '\n'
            text_fit_result += 'ndf = ' + str(np.round(self.fit_chi2_ndof[which_hist][1], precision)) + '\n'


        ax = axis
        if not axis:
            plt.figure()
            ax = plt

        ax.errorbar(self.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]],
                    self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]],
                    yerr=self.errors[which_hist][slice_list[0]:slice_list[1]:slice_list[2]],
                    fmt='o' + color, label=self.label)
        if show_fit:
            reduced_axis = self.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]]
            fit_axis = np.arange(reduced_axis[0], reduced_axis[-1], float(reduced_axis[1] - reduced_axis[0]) / 10)
            # ax.plot(fit_axis, self.fit_function(self.fit_result[which_hist][:,0], fit_axis), label='fit',color=color)
            reduced_func = self.fit_function
            if type(config).__name__ == 'ndarray':
                reduced_func = lambda p, x: self.fit_function(p, x, config=config[which_hist])
            ax.plot(fit_axis, reduced_func(self.fit_result[which_hist][:, 0], fit_axis), label='fit', color='r')
            ax.text(x_text, y_text, text_fit_result)
        if not axis:
            ax.xlabel(self.xlabel)
            ax.ylabel(self.ylabel)
        else:
            ax.set_yscale(scale)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.xaxis.get_label().set_ha('right')
            ax.xaxis.get_label().set_position((1, 0))
            ax.yaxis.get_label().set_ha('right')
            ax.yaxis.get_label().set_position((0, 1))

        if set_ylim:
            if not axis:
                ax.ylim(0, (np.max(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]) +
                            self.errors[
                    which_hist + (np.argmax(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]),)]) * 1.3)
            else:
                if scale != 'log':
                    ax.set_ylim(0,
                                (np.max(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]) +
                                 self.errors[
                                    which_hist + (np.argmax(
                                        self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]),)]) * 1.3)
                else:
                    ax.set_ylim(1e-1,
                                (np.max(self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]) +
                                 self.errors[
                                    which_hist + (np.argmax(
                                        self.data[which_hist][slice_list[0]:slice_list[1]:slice_list[2]]),)]) * 10)
        ax.legend(loc='best')
