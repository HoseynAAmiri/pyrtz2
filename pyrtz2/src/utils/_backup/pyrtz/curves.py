import pickle
import pandas as pd
from plotly import graph_objs as go
from plotly import offline as py
import numpy as np
import json
import ast
import scipy.optimize
from plotly import express as px
import io
import PyPDF2 as pdf

dark_colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
               'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
               'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
               'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
               'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

light_colors = ['rgba(31, 119, 180, .2)', 'rgba(255, 127, 14, .2)',
                'rgba(44, 160, 44, .2)', 'rgba(214, 39, 40, .2)',
                'rgba(148, 103, 189, .2)', 'rgba(140, 86, 75, .2)',
                'rgba(227, 119, 194, .2)', 'rgba(127, 127, 127, .2)',
                'rgba(188, 189, 34, .2)', 'rgba(23, 190, 207, .2)']


class Curve:
    '''A class representing a single force curve'''

    contact_index: int = None
    approach_range: list[int] = []
    dwell_range: list[int] = []
    k: float = None
    invOLS = None
    data: pd.DataFrame = None
    parameters: pd.DataFrame = None
    cols: dict[str, float] = None
    stiff_fit: dict[pd.DataFrame, float] = None
    biexponential_fit: dict[str] = None
    exponential_fit: dict = None

    def __init__(self, filename, data, parameters, z_col, t_col, f_col, ind_col, invOLS, k, dwell_range):
        '''Construct a new pyrtz.curves.Curve object. This method should not be called
        directly by the end user. In normal use, Curve objects should be created by calling
        pyrtz.asylum.load_ibw or pyrtz.asylum.load_curveset_ibw

        --------------------Arguments--------------------
        filename: A string containing the original .ibw
        file's name

        data: A pandas.DataFrame containing the force
        curve itself

        parameters: A dict containing additional
        information from the .ibw file's notes section

        z_col: The (string) name of the column in data
        which contains z position data

        t_col: The (string) name of the column in data
        which contains time data

        f_col: The (string) name of the column in data
        which contains force data

        ind_col: The (string) name of the column in data
        which contains the indentation data

        invOLS: The inverse optical lever sensitivity
        of the lever used for this measurement

        k: The spring constant of the lever used for
        this measurement

        dwell_range: A two entry list containing the row
        indices at which the dwell region begins and ends

        ---------------------Returns---------------------
        A new pyrtz.curves.Curve object'''

        self.dwell_range = dwell_range
        self.k = k
        self.invOLS = invOLS
        self.data = data
        self.parameters = parameters
        self.cols = {'z': z_col, 't': t_col, 'f': f_col, 'ind': ind_col}
        self.filename = filename

    def get_approach(self) -> pd.DataFrame:
        '''Get the approach section of the force curve

        ---------------------Returns---------------------
        A pandas.DataFrame containing the approach
        region of the force curve'''

        return self.data.loc[0:self.dwell_range[0], :]

    def get_dwell(self):
        '''Get the dwell section of the force curve

        ---------------------Returns---------------------
        A pandas.DataFrame containing the dwellregion of
        the force curve'''

        return self.data.loc[self.dwell_range[0]:self.dwell_range[1], :]

    def get_retract(self) -> pd.DataFrame:
        '''Get the retract section of the force curve

        ---------------------Returns---------------------
        A pandas.DataFrame containing the retract region
        of the force curve'''

        return self.data.loc[self.dwell_range[1]:, :]

    def set_contact_index(self, cp):
        '''Set the row index corresponding to this curve's
        contact point. Usually the end user should not
        call this function directly, instead, the contact
        points should be identified using pyrtz.annocp

        --------------------Arguments--------------------
        cp: the index in self.data which corresponds to
        the contact point for this force curve

        ---------------------Returns---------------------

        None'''

        self.contact_index = cp

    def correct_virt_defl(self):
        '''Correct for non-zero slope of approach curves
        before contact point. This function fits a line
        to any points before the annotated contact point
        and then subtracts this line from the approach
        curve. This function updates the curve object
        in place. Note: this function will only update
        the 'f' column of self.data.

        ---------------------Returns---------------------

        None'''

        if self.contact_index == None:
            raise Exception(
                'Contact index has not been set. Please update contact point annotations')

        fit_data = self.data.iloc[:self.contact_index, :]
        def model_func(x, m, b): return m*x+b
        x_data = fit_data.loc[:, self.cols['z']].to_numpy()
        y_data = fit_data.loc[:, self.cols['f']].to_numpy()
        popt, pconv = scipy.optimize.curve_fit(model_func, x_data, y_data)

        f_line = model_func(self.data.loc[:, self.cols['z']], *popt)

        self.data.loc[:, self.cols['f']
                      ] = self.data.loc[:, self.cols['f']]-f_line

    def fit_stiffness(self, probe_diameter, fit_range=[0, 1]):
        '''Fit this force curve using the hertz contact model
        for an elastic sphere indenting an elastic half space

        https://en.wikipedia.org/wiki/Contact_mechanics#Contact_between_a_sphere_and_a_half-space

        --------------------Arguments--------------------

        probe_diameter: The diameter of the indenting
        sphere, in meters

        fit_range: A two entry sequence defining the
        portion of the force curve to fit as a ratio
        of the maximum force

        [0,1] fits the entire curve

        [0.25, 0.75] fits the middle 50% of the curve

        any two numbers between 0 and 1 are acceptable as
        long as the second number is larger than the
        first

        ---------------------Returns---------------------

        None'''

        if self.contact_index == None:
            raise Exception(
                'Contact index has not been set, stiffness fits cannot continue')

        r = probe_diameter/2
        # Get only contact region and adjust force and indentation so values at contact are 0
        indent_raw = self.get_approach(
        ).loc[self.contact_index:, self.cols['ind']].to_numpy()
        force_raw = self.get_approach(
        ).loc[self.contact_index:, self.cols['f']].to_numpy()
        indent_norm = indent_raw-indent_raw[0]
        force_norm = force_raw-force_raw[0]

        def get_force(indentation, e_star):
            return (4/3)*e_star*(r**0.5)*(indentation**1.5)

        # Figure out data subset for fitting
        fmin = force_norm[-1]*fit_range[0]
        fmax = force_norm[-1]*fit_range[1]

        imin = 0
        imax = 0
        # iterate through force readings backwards (trigger first)
        for i in list(range(len(force_norm)))[::-1]:
            if force_norm[i] >= fmin:
                imin = i
            if force_norm[i] >= fmax:
                imax = i

        indent_norm_fit = indent_norm[imin:imax]
        force_norm_fit = force_norm[imin:imax]

        # run fit
        popt, pcov = scipy.optimize.curve_fit(
            get_force, indent_norm_fit, force_norm_fit)

        estar_fit = popt[0]
        fit_curve = pd.DataFrame(dict(
            ind=indent_norm_fit+indent_raw[0], f=get_force(indent_norm_fit, estar_fit)+force_raw[0]))
        self.stiff_fit = dict(curve=fit_curve, estar=estar_fit)

    def get_stiffness_fit_figure(self):
        '''Get a figure illustrating the fit resulting from
        the last call to self.fit_stiffness

        ---------------------Returns---------------------
        A plotly.graph_objs._figure.Figure object
        illustrating the current fit'''

        if not self.stiff_fit:
            raise Exception(
                'No stiffness fit has yet been performed. Run fit_stiffness method')

        measured_curve = self.get_approach().rename(columns={
            self.cols['z']: 'z', self.cols['ind']: 'ind', self.cols['t']: 't', self.cols['f']: 'f'})
        measured_curve.loc[:, 'curve'] = 'measured'
        fit_curve = self.stiff_fit['curve'].copy()
        fit_curve.loc[:, 'curve'] = 'fit'
        all_curves = pd.concat([measured_curve, fit_curve], ignore_index=True)
        fig = px.scatter(all_curves, x='ind', y='f', color='curve')
        fig.update_xaxes(title={'text': 'Indentation (m)'})
        fig.update_yaxes(title={'text': 'Force (N)'})
        fig.add_vline(x=measured_curve.loc[self.contact_index, 'ind'])
        return fig

    def fit_biexponential(self):
        '''Fit a biexponential decay function to the dwell region of this force curve

        ---------------------Returns---------------------

        None'''

        fit_data = self.get_dwell().rename(columns={
            self.cols['z']: 'z', self.cols['ind']: 'ind', self.cols['t']: 't', self.cols['f']: 'f'})
        f_raw = fit_data['f'].to_numpy()
        f0 = f_raw[0]
        t_raw = fit_data['t'].to_numpy()
        # adjust time to start at zero when the dwell begins
        t_norm = t_raw-t_raw[0]

        # make some good initial guesses
        c_guess = f_raw[-1]
        # force value corresponding to ~63% relaxation
        e_threshold = f0-0.63*(f0-c_guess)
        # corresponding time
        e_time = fit_data.loc[fit_data.loc[:, 'f'] < e_threshold, 't'].to_numpy()[
            0]
        tau1_guess = e_time
        tau2_guess = 0.1*tau1_guess
        a_guess = 0.4  # arbitrary

        def calc_force(t, tau1, tau2, A, C):
            return (f0-C)*(A*np.exp(-1*t*tau1)+(1-A)*np.exp(-1*t*tau2))+C

        bounds = ([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf])
        p0 = [tau1_guess, tau2_guess, a_guess, c_guess]

        popt, pconv = scipy.optimize.curve_fit(
            calc_force, t_norm, f_raw, bounds=bounds, p0=p0, jac='3-point')
        biexponential_fit = dict(
            tau1=popt[0], tau2=popt[1], A=popt[2], C=popt[3])

        fit_curve = pd.DataFrame(dict(t=fit_data['t'], f=calc_force(
            t_norm, biexponential_fit['tau1'], biexponential_fit['tau2'], biexponential_fit['A'], biexponential_fit['C'])))

        biexponential_fit['curve'] = fit_curve
        biexponential_fit['tau_fast'] = max(
            biexponential_fit['tau1'], biexponential_fit['tau2'])
        biexponential_fit['tau_slow'] = min(
            biexponential_fit['tau1'], biexponential_fit['tau2'])
        self.biexponential_fit = biexponential_fit

    def get_biexponential_fit_figure(self):
        '''Get a figure illustrating the fit resulting from
        the last call to self.fit_biexponential

        ---------------------Returns---------------------
        A plotly.graph_objs._figure.Figure object
        illustrating the current fit'''

        if not self.biexponential_fit:
            raise Exception(
                'No biexponential fit has yet been performed. Run fit_biexponential method')

        measured_curve = self.get_dwell().rename(columns={
            self.cols['z']: 'z', self.cols['ind']: 'ind', self.cols['t']: 't', self.cols['f']: 'f'})
        measured_curve.loc[:, 'curve'] = 'measured'

        fit_curve = self.biexponential_fit['curve'].copy()
        fit_curve.loc[:, 'curve'] = 'fit'

        all_curves = pd.concat([measured_curve, fit_curve], ignore_index=True)
        fig = px.scatter(all_curves, x='t', y='f', color='curve')
        fig.update_xaxes(title={'text': 'Time (s)'})
        fig.update_yaxes(title={'text': 'Force (N)'})
        return fig

    def fit_exponential(self):
        '''Fit an exponential decay function to the dwell region of this force curve

        ---------------------Returns---------------------

        None'''

        fit_data = self.get_dwell().rename(columns={
            self.cols['z']: 'z', self.cols['ind']: 'ind', self.cols['t']: 't', self.cols['f']: 'f'})
        f_raw = fit_data['f'].to_numpy()
        f0 = f_raw[0]
        t_raw = fit_data['t'].to_numpy()
        # adjust time to start at zero when the dwell begins
        t_norm = t_raw-t_raw[0]

        # make some good initial guesses
        c_guess = f_raw[-1]
        # force value corresponding to ~63% relaxation
        e_threshold = f0-0.63*(f0-c_guess)
        # corresponding time
        e_time = fit_data.loc[fit_data.loc[:, 'f'] < e_threshold, 't'].to_numpy()[
            0]
        tau0_guess = 1/e_time

        def calc_force(t, tau0, C):
            return (f0-C)*np.exp(-1*t*tau0)+C

        bounds = ([0, -np.inf], [np.inf, np.inf])
        p0 = [tau0_guess, c_guess]

        popt, pconv = scipy.optimize.curve_fit(
            calc_force, t_norm, f_raw, bounds=bounds, p0=p0, jac='3-point')
        exponential_fit = dict(tau0=popt[0], C0=popt[1])

        fit_curve = pd.DataFrame(dict(t=fit_data['t'], f=calc_force(
            t_norm, exponential_fit['tau0'], exponential_fit['C0'])))

        exponential_fit['curve'] = fit_curve

        self.exponential_fit = exponential_fit

    def get_exponential_fit_figure(self):
        '''Get a figure illustrating the fit resulting from
        the last call to self.fit_exponential

        ---------------------Returns---------------------
        A plotly.graph_objs._figure.Figure object
        illustrating the current fit'''

        if not self.exponential_fit:
            raise Exception(
                'No exponential fit has yet been performed. Run fit_exponential method')

        measured_curve = self.get_dwell().rename(columns={
            self.cols['z']: 'z', self.cols['ind']: 'ind', self.cols['t']: 't', self.cols['f']: 'f'})
        measured_curve.loc[:, 'curve'] = 'measured'

        fit_curve = self.exponential_fit['curve'].copy()
        fit_curve.loc[:, 'curve'] = 'fit'

        all_curves = pd.concat([measured_curve, fit_curve], ignore_index=True)
        fig = px.scatter(all_curves, x='t', y='f', color='curve')
        fig.update_xaxes(title={'text': 'Time (s)'})
        fig.update_yaxes(title={'text': 'Force (N)'})
        return fig


class CurveSet:
    '''An object representing a set of force curves'''

    ident_labels = None
    curve_dict = None

    def __init__(self, ident_labels, curve_dict):
        '''Construct a new pyrtz.curves.CurveSet object
        This constructor should not usually be called
        by an end user. Instead use
        pyrtz.asylum.load_curveset_ibw

        --------------------Arguments--------------------

        ident_labels: A list containing labels
        corresponding to unique curve identifiers, such
        as those passed as the ident_labels argument of
        pyrtz.asylum.load_curveset_ibw

        curve_dict: A dict whose keys are unique
        identifiers and whose values are
        pyrtz.curves.Curve objects

        ---------------------Returns---------------------

        A new pyrtz.curves.CurveSet object'''

        self.ident_labels = ident_labels
        self.curve_dict = curve_dict

    def __iter__(self):
        '''Enable iteration over CurveSets'''

        return self.curve_dict.__iter__()

    def pickle(self, filename):
        '''Dump this curveset to a file

        --------------------Arguments--------------------

        filename: The path where the object should be
        saved.

        ---------------------Returns---------------------

        None'''

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def keys(self) -> list:
        '''Return a list of the unique identifiers
        corresponding to each curve in the CurveSet

        ---------------------Returns---------------------

        A list of unique curve identifiers'''

        return list(self.curve_dict.keys())

    def __getitem__(self, index) -> Curve:
        '''Enable indexing of CurveSets using
        curve_set[key] syntax

        ---------------------Returns---------------------

        A single pyrtz.curves.CurveSet object'''

        return self.curve_dict[index]

    def remove_curve(self, key):
        '''Drop a curve from the CurveSet

        --------------------Arguments--------------------

        key: the unique identifier of the curve to drop

        ---------------------Returns---------------------

        None'''

        del self.curve_dict[key]

    def remove_unannotated(self):
        '''Drop all curves for which their contact point
        has not been annotated (all curves for which
        curve.contact_index==0)

        --------------------Arguments--------------------

        None

        ---------------------Returns---------------------

        None'''

        idents_to_drop = []
        for ident in self:
            if self[ident].contact_index == 0:
                idents_to_drop.append(ident)

        for i in idents_to_drop:
            self.remove_curve(i)

    def correct_virt_defl(self):
        '''Correct for non-zero slope of approach curves
        before contact point. This function fits a line
        to any points before the annotated contact point
        and then subtracts this line from the approach
        curve. This function updates the curve object
        in place. Note: this function will only update
        the 'f' column of curve.data.

        ---------------------Returns---------------------

        None'''

        for key in self:
            self[key].correct_virt_defl()

    def collate_curves(self) -> pd.DataFrame:
        '''Return all the force curves contained in this
        CurveSet as a single pandas.DataFrame

        ---------------------Returns---------------------

        A pandas.DataFrame containing every force curve
        in the CurveSet'''

        all_curves = []
        for ident in self.keys():
            this_curve = self[ident].data.copy()
            for label, value in zip(self.ident_labels, ident):
                this_curve.loc[:, label] = value
            all_curves.append(this_curve)
        return pd.concat(all_curves, ignore_index=True)

    # def normalize_curves(curves,idents,t_col='t',z_col='zSensr',f_col='force'):
    def normalize_curves(self) -> pd.DataFrame:
        '''Normalize all curves so that the trigger point
        corresponds to t=0, z=0, ind=0, f=0 and return all the
        resulting normalized force curves as a single
        pandas.DataFrame

        ---------------------Returns---------------------

        A pandas.DataFrame containing every force curve
        in the CurveSet normalized so that the trigger
        point corresponds to t=0, z=0, f=0'''

        curves = self.collate_curves()
        idents = list(self.ident_labels)
        cols = self[self.keys()[0]].cols
        t_col = cols['t']
        z_col = cols['z']
        ind_col = cols['ind']
        f_col = cols['f']
        desired_cols = [*idents, t_col, z_col, ind_col, f_col]
        curves = curves.loc[:, desired_cols]

        maxf = curves.loc[:, [*idents, f_col]
                          ].groupby(idents).agg("max").reset_index()
        maxf_cols = list(maxf.columns)
        for i in range(len(maxf_cols)):
            if maxf_cols[i] == f_col:
                maxf_cols[i] = 'max_f'

        maxf.columns = maxf_cols

        curves = pd.merge(curves, maxf)
        norm_vals = curves.query(f'{f_col}==max_f')
        norm_vals = norm_vals.loc[:, [*idents, t_col, ind_col, z_col]]
        norm_cols = list(norm_vals.columns)
        for j in range(len(norm_cols)):
            if norm_cols[j] == t_col:
                norm_cols[j] = 't0'
            elif norm_cols[j] == ind_col:
                norm_cols[j] = 'ind0'
            elif norm_cols[j] == z_col:
                norm_cols[j] = 'z0'

        norm_vals.columns = norm_cols

        curves = pd.merge(curves, norm_vals)
        curves.loc[:, 't_norm'] = curves.loc[:, t_col]-curves.loc[:, 't0']
        curves.loc[:, 'z_norm'] = curves.loc[:, z_col]-curves.loc[:, 'z0']
        curves.loc[:, 'ind_norm'] = curves.loc[:,
                                               ind_col]-curves.loc[:, 'ind0']
        curves.loc[:, 'f_norm'] = curves.loc[:, f_col]-curves.loc[:, 'max_f']

        curves = curves.loc[:, [*idents, 't_norm',
                                'z_norm', 'ind_norm', 'f_norm']]

        return curves

    def plot_traj(self, group, filename='characteristic_trajectories.html', round_dec=4):
        '''Plot the characteristic force curves for each
        unique value of group.

        --------------------Arguments--------------------

        group: The ident_label for which unique values
        correspond to different experimental conditions
        to be plotted

        filename: The filepath where the resulting plot
        should be stored

        round_dec: The number of decimal places to round
        the times of each sample to

        ---------------------Returns---------------------

        None'''

        data = self.normalize_curves()
        time_col = 't_norm'
        f_col = 'f_norm'
        data.loc[:, group] = [str(a) for a in data.loc[:, group]]
        data = data.loc[:, [group, time_col, f_col]]
        data.loc[:, time_col] = [np.round(a, round_dec)
                                 for a in data.loc[:, time_col]]
        upper = data.groupby([group, time_col]).agg(
            lambda x: np.quantile(x, .75)).reset_index()
        median = data.groupby([group, time_col]).agg(
            lambda x: np.quantile(x, .5)).reset_index()
        lower = data.groupby([group, time_col]).agg(
            lambda x: np.quantile(x, .25)).reset_index()

        upper.loc[:, 'metric'] = 'upper'
        median.loc[:, 'metric'] = 'median'
        lower.loc[:, 'metric'] = 'lower'

        all_metrics = pd.concat([upper, median, lower], ignore_index=True)

        traces = []
        all_groups = list(set(all_metrics.loc[:, group]))
        num_reps = np.ceil(len(all_groups)/len(dark_colors))
        for g, dark_c, light_c in zip(all_groups, int(num_reps)*dark_colors, int(num_reps)*light_colors):
            this_group = all_metrics.loc[all_metrics.loc[:, group] == g, :]
            this_median = this_group.loc[this_group.loc[:,
                                                        'metric'] == 'median']
            this_upper = this_group.loc[this_group.loc[:, 'metric'] == 'upper']
            this_lower = this_group.loc[this_group.loc[:, 'metric'] == 'lower']
            main_trace = go.Scatter(x=this_median[time_col],
                                    y=this_median[f_col],
                                    line=dict(color=dark_c),
                                    mode='lines',
                                    name=g+' median')
            error_trace = go.Scatter(x=list(this_upper[time_col])+list(this_lower[time_col])[::-1],
                                     y=list(this_upper[f_col]) +
                                     list(this_lower[f_col])[::-1],
                                     fill='toself',
                                     fillcolor=light_c,
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip",
                                     name=g+' error')
            traces.extend([main_trace, error_trace])

        fig = go.Figure(traces)

        py.plot(fig, filename=filename)
        return all_metrics

    def update_cp_annotations(self, cp_dict):
        '''Update the stored contact point for
        every curve in the CurveSet. Normally end
        users should instead pass the file created by
        pyrtz.annocp to update_cp_annotations_from_file
        instead

        --------------------Arguments--------------------

        cp_dict: A dict where each key is a unique curve
        identifier, for which the corresponding value is
        the index of the contact point in the backing
        pandas.DataFrame

        ---------------------Returns---------------------

        None'''

        for key in cp_dict:
            self[key].set_contact_index(cp_dict[key])

    def update_cp_annotations_from_file(self, cp_file):
        '''Update the stored contact point for every
        curve in the CurveSet using a .json file created
        by pyrtz.annocp

        --------------------Arguments--------------------

        cp_file: A .json file containing contact point
        annotations created by pyrtz.annocp

        ---------------------Returns---------------------

        None'''

        with open(cp_file, 'rt') as cf:
            anno_str_dict = json.load(cf)

        anno_tuple_dict = {}
        for key in anno_str_dict:
            tuple_key = ast.literal_eval(key)
            anno_tuple_dict[tuple_key] = anno_str_dict[key]

        self.update_cp_annotations(anno_tuple_dict)

    def fit_all_stiff(self, probe_diameter, fit_range=[0, 1]):
        '''Fit all force curves in this CurveSet using the
        hertz contact model for an elastic sphere
        indenting an elastic half space

        https://en.wikipedia.org/wiki/Contact_mechanics#Contact_between_a_sphere_and_a_half-space

        --------------------Arguments--------------------

        probe_diameter: The diameter of the indenting
        sphere, in meters

        fit_range: A two entry sequence defining the
        portion of the force curve to fit as a ratio
        of the maximum force

        [0,1] fits the entire curve

        [0.25, 0.75] fits the middle 50% of the curve

        any two numbers between 0 and 1 are acceptable as
        long as the second number is larger than the
        first

        ---------------------Returns---------------------

        None'''

        for key in self:
            self[key].fit_stiffness(probe_diameter, fit_range)

    def fit_all_biexponential(self):
        '''Fit the dwell region of every curve contained in
        this CurveSet to a biexponential decay function

        ---------------------Returns---------------------

        None'''

        for key in self:
            self[key].fit_biexponential()

    def fit_all_exponential(self):
        '''Fit the dwell region of every curve contained in
        this CurveSet to an exponential decay function

        ---------------------Returns---------------------

        None'''

        for key in self:
            self[key].fit_exponential()

    def fit_all(self, probe_diameter, fit_range=[0, 1]):
        '''Fit this force curve using the hertz contact model
        for an elastic sphere indenting an elastic half space
        and then fit the dwell region of each curve contained
        in this CurveSet to a biexponential and exponential
        decay function

        https://en.wikipedia.org/wiki/Contact_mechanics#Contact_between_a_sphere_and_a_half-space

        --------------------Arguments--------------------

        probe_diameter: The diameter of the indenting
        sphere, in meters

        fit_range: A two entry sequence defining the
        portion of the force curve to fit as a ratio
        of the maximum force

        [0,1] fits the entire curve

        [0.25, 0.75] fits the middle 50% of the curve

        any two numbers between 0 and 1 are acceptable as
        long as the second number is larger than the
        first

        ---------------------Returns---------------------

        None'''

        self.fit_all_stiff(probe_diameter, fit_range)
        self.fit_all_biexponential()
        self.fit_all_exponential()

    def get_stiff_results(self) -> pd.DataFrame:
        '''Export the results of fit_all_stiff as a
        pandas.DataFrame. If fit_all_stiff has not yet
        been called on this CurveSet this method will
        raise an exception

        ---------------------Returns---------------------

        A pandas.DataFrame containing the fit parameters
        for every Curve in this CurveSet'''

        entries = []
        for key in self:
            df_dict = {}
            for label, ident in zip(self.ident_labels, key):
                df_dict[label] = [ident]
            df_dict['estar'] = [self[key].stiff_fit['estar']]
            entries.append(pd.DataFrame(df_dict))
        return (pd.concat(entries, ignore_index=True))

    def get_biexponential_results(self) -> pd.DataFrame:
        '''Export the results of fit_all_biexponential
        as a pandas.DataFrame. If fit_all_biexponential
        has not yet been called on this CurveSet, this
        method will raise an exception

        ---------------------Returns---------------------

        A pandas.DataFrame containing the fit parameters
        for every Curve in this CurveSet'''

        entries = []
        for key in self:
            df_dict = {}
            for label, ident in zip(self.ident_labels, key):
                df_dict[label] = [ident]
            df_dict['tau1'] = [self[key].biexponential_fit['tau1']]
            df_dict['tau2'] = [self[key].biexponential_fit['tau2']]
            df_dict['tau_fast'] = [self[key].biexponential_fit['tau_fast']]
            df_dict['tau_slow'] = [self[key].biexponential_fit['tau_slow']]
            df_dict['A'] = [self[key].biexponential_fit['A']]
            df_dict['C'] = [self[key].biexponential_fit['C']]
            entries.append(pd.DataFrame(df_dict))
        return (pd.concat(entries, ignore_index=True))

    def get_exponential_results(self) -> pd.DataFrame:
        '''Export the results of fit_all_exponential
        as a pandas.DataFrame. If fit_all_exponential
        has not yet been called on this CurveSet, this
        method will raise an exception

        ---------------------Returns---------------------

        A pandas.DataFrame containing the fit parameters
        for every Curve in this CurveSet'''

        entries = []
        for key in self:
            df_dict = {}
            for label, ident in zip(self.ident_labels, key):
                df_dict[label] = [ident]
            df_dict['tau0'] = [self[key].exponential_fit['tau0']]
            df_dict['C0'] = [self[key].exponential_fit['C0']]
            entries.append(pd.DataFrame(df_dict))
        return (pd.concat(entries, ignore_index=True))

    def get_all_results(self) -> pd.DataFrame:
        '''Export the results of all fits which have been
        performed on this CurveSet

        ---------------------Returns---------------------

        A pandas.DataFrame containing the fit parameters
        for every Curve in this CurveSet'''
        all_results = []
        try:
            stiff_results = self.get_stiff_results()
            all_results.append(stiff_results)
        except TypeError:  # This is the exception that happens when the fits have not yet been run
            pass
        try:
            biexponential_results = self.get_biexponential_results()
            all_results.append(biexponential_results)
        except TypeError:  # This is the exception that happens when the fits have not yet been run
            pass
        try:
            exponential_results = self.get_exponential_results()
            all_results.append(exponential_results)
        except TypeError:  # This is the exception that happens when the fits have not yet been run
            pass
        if len(all_results) < 1:
            raise Exception(
                'No fits have yet been run. Please run a fit before attempting to export results')

        elif len(all_results) == 1:
            return all_results[0]
        else:
            res = all_results.pop(0)
            while all_results:
                res = pd.merge(res, all_results.pop(0))
        return res

    def export_stiffness_fit_report(self, filepath):
        '''Create a .pdf document displaying all
        stiffness fits for this CurveSet

        --------------------Arguments--------------------

        filepath: The path where the fit report should be
        saved

        ---------------------Returns---------------------

        None'''

        merger = pdf.PdfMerger()
        for key in self:
            this_curve = self[key]
            if this_curve.stiff_fit == None:
                raise Exception(
                    'Stiffness fit has not been performed, please run stiffness fit before attempting to export fit reports')
            title = ''
            for label, ident in zip(self.ident_labels, key):
                title = title+f'{label}{ident}'

            title = title+f" estar: {this_curve.stiff_fit['estar']}Pa"
            this_fit_fig = this_curve.get_stiffness_fit_figure()
            this_fit_fig.update_layout(title={'text': title})

            this_fit_fig_pdf = io.BytesIO(this_fit_fig.to_image(format='pdf'))
            merger.append(this_fit_fig_pdf)

        merger.write(filepath)

    def export_biexponential_fit_report(self, filepath):
        '''Create a .pdf document displaying all
        biexponential fits for this CurveSet

        --------------------Arguments--------------------

        filepath: The path where the fit report should be
        saved

        ---------------------Returns---------------------

        None'''

        merger = pdf.PdfMerger()
        for key in self:
            this_curve = self[key]
            if this_curve.biexponential_fit == None:
                raise Exception(
                    'Biexponential fit has not been performed, please run biexponential fit before attempting to export fit reports')
            title = ''
            for label, ident in zip(self.ident_labels, key):
                title = title+f'{label}{ident}'

            title = title + \
                f" tau_fast: {this_curve.biexponential_fit['tau_fast']}/s, tau_slow:{this_curve.biexponential_fit['tau_slow']}/s"
            this_fit_fig = this_curve.get_biexponential_fit_figure()
            this_fit_fig.update_layout(title={'text': title})

            this_fit_fig_pdf = io.BytesIO(this_fit_fig.to_image(format='pdf'))
            merger.append(this_fit_fig_pdf)

        merger.write(filepath)

    def export_exponential_fit_report(self, filepath):
        '''Create a .pdf document displaying all
        exponential fits for this CurveSet

        --------------------Arguments--------------------

        filepath: The path where the fit report should be
        saved

        ---------------------Returns---------------------

        None'''

        merger = pdf.PdfMerger()
        for key in self:
            this_curve = self[key]
            if this_curve.exponential_fit == None:
                raise Exception(
                    'Biexponential fit has not been performed, please run biexponential fit before attempting to export fit reports')
            title = ''
            for label, ident in zip(self.ident_labels, key):
                title = title+f'{label}{ident}'

            title = title+f" tau0: {this_curve.exponential_fit['tau0']}/s"
            this_fit_fig = this_curve.get_exponential_fit_figure()
            this_fit_fig.update_layout(title={'text': title})

            this_fit_fig_pdf = io.BytesIO(this_fit_fig.to_image(format='pdf'))
            merger.append(this_fit_fig_pdf)

        merger.write(filepath)
