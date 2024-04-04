from plotly import graph_objs as go
import plotly.express as px

import PyPDF2 as pdf
import pickle
import pandas as pd
import numpy as np
import json
import ast
import io
from tqdm import tqdm
from typing import Any, Iterable

from .fit import lin_fit, poly_fit, hertzian_fit, biexponential_fit
from .src.components.fig import make_fig


def read_json_file(file: str) -> dict[tuple, bool | int]:
    with open(file, 'rt') as cf:
        anno_str_dict = json.load(cf)

    anno_tuple_dict = {}
    for key, value in anno_str_dict.items():
        tuple_key = ast.literal_eval(key)
        anno_tuple_dict[tuple_key] = value

    return anno_tuple_dict


class Curve:

    _data: pd.DataFrame
    corrected: bool = False
    contact_index: int = 0
    contact_values: pd.Series | None = None
    max_ind: float
    max_f: float
    vel_ind: float
    vel_z: float
    figs_data: dict[str, tuple[np.ndarray, np.ndarray]]
    fits_data: dict[str, tuple[np.ndarray, np.ndarray]]
    indent_param: list[float]
    indent_R2: float
    hertzian_param: list[float]
    hertzian_R2: float
    dwell_param: list[float]
    dwell_R2: float
    relaxation_param: list[float]
    relaxation_R2: float
    contact_fig: go.Figure
    dwell_relax_fig: go.Figure

    def __init__(
            self,
            filename: str,
            data: pd.DataFrame,
            notes: dict,
            invOLS: float,
            k: float,
            dwell_range: list[int],
    ) -> None:

        self.filename = filename
        self.data = data
        self.notes = notes
        self.invOLS = invOLS
        self.k = k
        self.dwell_range = dwell_range

        self.backup_data()

    def backup_data(self) -> None:
        self._data = self.data.copy()

    def restore_data(self) -> None:
        self.data = self._data.copy()

    def reduce_data(self) -> None:
        self.data.drop(['rawz', 'defl'], axis=1, inplace=True)

    def set_contact_index(self, cp: int) -> None:
        self.contact_index = cp

    def check_contact(self) -> None:
        if self.contact_index == 0:
            raise Exception('Contact index has not been set.')

    def get_contact_values(self) -> pd.Series:
        self.check_contact()
        if self.contact_values is None:
            self.contact_values = self.data.iloc[self.contact_index].copy()
        return self.contact_values

    def adjust_to_contact(self) -> None:
        self.data -= self.get_contact_values()

    def get_data_between(self, start: int, last: int) -> pd.DataFrame:
        return self.data.iloc[start:last].reset_index(drop=True)

    def get_approach(self) -> pd.DataFrame:
        return self.get_data_between(0, self.dwell_range[0])

    def get_preapproach(self) -> pd.DataFrame:
        return self.get_data_between(0, self.contact_index)

    def get_indent(self) -> pd.DataFrame:
        return self.get_data_between(self.contact_index, self.dwell_range[0])

    def get_indent_until(self, ind: float) -> pd.DataFrame:
        indent = self.get_indent()
        ind_index = (np.abs(indent['ind'].to_numpy() - ind)).argmin()
        return indent.iloc[:ind_index].reset_index(drop=True)

    def get_indent_between(self, start: float, last: float) -> pd.DataFrame:
        indent = self.get_indent()
        start_index = int(start * len(indent))
        last_index = int(last * len(indent))
        return indent.iloc[start_index:last_index].reset_index(drop=True)

    def get_dwell(self) -> pd.DataFrame:
        return self.get_data_between(self.dwell_range[0], self.dwell_range[1])

    def get_relaxation(self) -> pd.DataFrame:
        dwell = self.get_dwell()
        max_ind = dwell['ind'].argmax()
        return dwell.iloc[:max_ind].reset_index(drop=True)

    def get_repulsion(self) -> pd.DataFrame:
        dwell = self.get_dwell()
        max_ind = dwell['ind'].argmax()
        return dwell.iloc[max_ind:].reset_index(drop=True)

    def get_retract(self) -> pd.DataFrame:
        return self.get_data_between(self.dwell_range[1], -1)

    def get_trigger_data(self) -> pd.Series:
        indent = self.get_indent()
        return indent.iloc[-1].reset_index(drop=True)

    def get_approach_rates(self) -> None:
        indent = self.get_indent()
        t, ind, z = indent['t'], indent['ind'], indent['z']
        self.vel_ind = lin_fit(t, ind)[0][0]
        self.vel_z = lin_fit(t, z)[0][0]

    def correct_virtual_defl(self) -> None:
        self.check_contact()
        if self.corrected:
            self.restore_data()

        preapproach = self.get_preapproach()
        x = preapproach['z'].to_numpy()
        y = preapproach['f'].to_numpy()
        popt, _, _ = lin_fit(x, y)
        f_line = np.poly1d(popt)(self.data['z'])
        self.data['f'] = self.data['f'] - f_line

    def detect_contact(self) -> int:
        current_contact_index = self.contact_index
        approach = self.get_approach()
        max_index = len(approach) - 1
        index = max_index
        while True:
            self.set_contact_index(index)
            self.correct_virtual_defl()
            approach_new = self.get_approach()
            force = approach_new['f'].to_numpy()
            mean = np.mean(force[:index+1])

            ratio = force[index] / force[-1]
            step_size = max(1, int((0.01 * max_index) ** ratio))
            index = index - step_size
            if force[index] < mean:
                contact_index = index + 1
                break

        self.set_contact_index(current_contact_index)
        self.restore_data()
        return contact_index

    def fit_indent(self) -> tuple:
        self.check_contact()
        indent = self.get_indent()
        return poly_fit(indent['ind'], indent['f'])

    def fit_indent_until(self, probe_diameter: float, ind: float) -> tuple:
        self.check_contact()
        indent_until = self.get_indent_until(ind)
        return hertzian_fit(indent_until['ind'], indent_until['f'], probe_diameter)

    def fit_indent_between(self, probe_diameter: float, interval: list[float] = [0.0, 1.0]) -> tuple:
        self.check_contact()
        indent_between = self.get_indent_between(interval[0], interval[1])
        return hertzian_fit(indent_between['ind'], indent_between['f'], probe_diameter)

    def fit_dwell(self) -> tuple:
        dwell = self.get_dwell()
        return biexponential_fit(dwell['t'], dwell['f'])

    def fit_relaxation(self) -> tuple:
        relaxation = self.get_relaxation()
        return biexponential_fit(relaxation['t'], relaxation['ind'])

    def get_figs_data(self, vd: bool = False, adjust: bool = False) -> None:
        if vd or adjust:
            self.restore_data()
        if vd:
            self.correct_virtual_defl()
        if adjust:
            self.adjust_to_contact()

        approach = self.get_approach()

        dwell = self.get_dwell()
        relaxation = self.get_relaxation()

        self.figs_data = {
            'approach': (approach['ind'].to_numpy(), approach['f'].to_numpy()),
            'dwell': (dwell['t'].to_numpy(), dwell['f'].to_numpy()),
            'relaxation': (relaxation['t'].to_numpy(), relaxation['ind'].to_numpy()),
        }

    def get_contact_fig_plot(self) -> go.Figure:
        fig = make_fig(
            title=fr"$\text{{Selected Contact Point: {self.contact_index}}}$",
            xaxis=r"$Indentation \text{ (m)}$"
        )

        x, y = self.figs_data['approach']
        hover_texts = [f'Index: {i}' for i in range(len(x))]
        trace = go.Scattergl(
            x=x,
            y=y,
            name='Approach',
            mode='markers',
            text=hover_texts,
            hoverinfo='text',
            hoverlabel={'bgcolor': 'red'},
            marker={'color': 'blue'},
        )

        fig.add_trace(trace)

        x_line = x[self.contact_index]
        y_line = y[self.contact_index]
        fig.add_vline(x=x_line, line=dict(color='red', width=1.2))
        fig.add_hline(y=y_line, line=dict(color='red', width=1.2))
        self.contact_fig = fig
        return fig

    def get_dwell_relax_fig_plot(self) -> go.Figure:
        fig = make_fig(
            title=r"$\text{Dwell and Relaxation}$",
            xaxis=r"$Time \text{ (s)}$"
        )

        fig.update_layout(
            yaxis2=dict(
                title=r"$Indentation \text{ (m)}$",
                overlaying='y',
                side='right'
            )
        )

        x, y = self.figs_data['dwell']
        trace = go.Scattergl(
            x=x,
            y=y,
            name='Dwell',
            mode='markers',
            marker={'color': 'red'},
        )

        fig.add_trace(trace)

        x, y = self.figs_data['relaxation']
        trace = go.Scattergl(
            x=x,
            y=y,
            yaxis='y2',
            name='Relaxation',
            mode='markers',
            marker={'color': 'blue'},
        )

        fig.add_trace(trace)

        self.dwell_relax_fig = fig
        return fig

    def get_fits_data(self, probe_diameter: float, ind: float | list[float]) -> None:
        indent = self.get_indent()
        indent_x_pred = indent['ind'].to_numpy()
        self.indent_param, self.indent_R2, indent_y_pred = self.fit_indent()

        self.max_ind = max(indent['ind'])
        self.max_f = max(indent['f'])

        if isinstance(ind, float):
            self.hertzian_param, self.hertzian_R2, hertzian_y_pred = self.fit_indent_until(
                probe_diameter, ind)
            hertzian_x_pred = self.get_indent_until(ind)['ind'].to_numpy()
        elif isinstance(ind, list) and len(ind) == 2:
            self.hertzian_param, self.hertzian_R2, hertzian_y_pred = self.fit_indent_between(
                probe_diameter, interval=ind)
            hertzian_x_pred = self.get_indent_between(ind[0], ind[1])[
                'ind'].to_numpy()

        dwell = self.get_dwell()
        dwell_x_pred = dwell['t'].to_numpy()
        self.dwell_param, self.dwell_R2, dwell_y_pred = self.fit_dwell()

        relaxation = self.get_relaxation()
        relaxation_x_pred = relaxation['t'].to_numpy()
        self.relaxation_param, self.relaxation_R2, relaxation_y_pred = self.fit_relaxation()

        self.fits_data = {
            'indent': (indent_x_pred, indent_y_pred),
            'hertzian': (hertzian_x_pred, hertzian_y_pred),
            'dwell': (dwell_x_pred, dwell_y_pred),
            'relaxation': (relaxation_x_pred, relaxation_y_pred),
        }

    def add_contact_fit(self) -> go.Figure:
        x, y = self.fits_data['indent']
        trace = go.Scattergl(
            x=x,
            y=y,
            name='PolyFit',
            mode='lines',
            line={'color': 'red'},
        )
        self.contact_fig.add_trace(trace)

        x, y = self.fits_data['hertzian']
        trace = go.Scattergl(
            x=x,
            y=y,
            name='HertzianFit',
            mode='lines',
            line={
                'color': 'green',
                'width': 3,
            },
        )
        self.contact_fig.add_trace(trace)

        return self.contact_fig

    def add_dwell_relax_fit(self) -> go.Figure:
        x, y = self.fits_data['dwell']

        trace = go.Scattergl(
            x=x,
            y=y,
            name='DwellFit',
            mode='lines',
            line={
                'color': 'green',
                'width': 3,
            },
        )
        self.dwell_relax_fig.add_trace(trace)

        x, y = self.fits_data['relaxation']
        trace = go.Scattergl(
            x=x,
            y=y,
            yaxis='y2',
            name='RelaxationFit',
            mode='lines',
            line={
                'color': 'green',
                'width': 3,
            },
        )
        self.dwell_relax_fig.add_trace(trace)

        return self.dwell_relax_fig

    def get_contact_fig_pdf(self) -> go.Figure:
        title = fr"$\text{{Selected Contact Point: {self.contact_index}}}$"
        xaxis = r"$Indentation \text{ (m)}$"
        fig = make_fig(title, xaxis)

        x, y = self.figs_data['approach']
        trace = go.Scatter(x=x, y=y, name='Approach', marker={'color': 'blue'})
        fig.add_trace(trace)

        x_line = x[self.contact_index]
        y_line = y[self.contact_index]
        fig.add_shape(type='line', x0=x_line, x1=x_line, y0=min(y),
                      y1=max(y), line=dict(color='red', width=1.2))
        fig.add_shape(type='line', y0=y_line, y1=y_line, x0=min(x),
                      x1=max(x), line=dict(color='red', width=1.2))

        if hasattr(self, 'fits_data'):
            x, y = self.fits_data['indent']
            trace = go.Scatter(x=x, y=y, name='PolyFit',
                               mode='lines', line={'color': 'red'})
            fig.add_trace(trace)

            x, y = self.fits_data['hertzian']
            trace = go.Scatter(x=x, y=y, name='HertzianFit',
                               mode='lines', line={'color': 'green', 'width': 3})
            fig.add_trace(trace)

        return fig

    def get_dwell_relax_fig_pdf(self) -> go.Figure:
        title = r"$\text{Dwell and Relaxation}$"
        xaxis = r"$Time \text{ (s)}$"
        fig = make_fig(title, xaxis)

        x, y = self.figs_data['dwell']
        trace = go.Scatter(x=x, y=y, name='Dwell',
                           marker_color='red')
        fig.add_trace(trace)

        fig.update_layout(
            yaxis2=dict(
                title=r"$Indentation \text{ (m)}$",
                overlaying='y',
                side='right',
                showgrid=False,
                ticks='outside',
                tickwidth=2,
                showline=True,
                linewidth=2,
                linecolor='black',
                tickprefix=r"$",
                ticksuffix=r"$",
            )
        )

        x, y = self.figs_data['relaxation']
        trace = go.Scatter(x=x, y=y, name='Relaxation',
                           marker_color='blue', yaxis='y2')
        fig.add_trace(trace)

        if hasattr(self, 'fits_data'):
            x, y = self.fits_data['dwell']
            trace = go.Scatter(x=x, y=y, name='DwellFit',
                               mode='lines', line={'color': 'green', 'width': 3})
            fig.add_trace(trace)

            x, y = self.fits_data['relaxation']
            trace = go.Scatter(x=x, y=y, yaxis='y2', name='RelaxationFit',
                               mode='lines', line={'color': 'green', 'width': 3})
            fig.add_trace(trace)

        return fig

    def get_all_fits(self) -> dict:
        fit_results_dict = {
            'max_ind': self.max_ind,
            'max_f': self.max_f,

            'vel_ind': self.vel_ind,
            'vel_z': self.vel_z,

            'indent_a': self.indent_param[0],
            'indent_b': self.indent_param[1],
            'indent_c': self.indent_param[2],
            'indent_R2': self.indent_R2,

            'Hertzian_E': self.hertzian_param[0],
            'Hertzian_R2': self.hertzian_R2,

            'dwell_c': self.dwell_param[0],
            'dwell_tau1': self.dwell_param[1],
            'dwell_tau2': self.dwell_param[2],
            'dwell_tauMax': max(self.dwell_param[1], self.dwell_param[2]),
            'dwell_tauMin': min(self.dwell_param[1], self.dwell_param[2]),
            'dwell_yf': self.dwell_param[3],
            'dwell_R2': self.dwell_R2,

            'relaxation_c': self.relaxation_param[0],
            'relaxation_tau1': self.relaxation_param[1],
            'relaxation_tau2': self.relaxation_param[2],
            'relaxation_tauMax': max(self.relaxation_param[1], self.relaxation_param[2]),
            'relaxation_tauMin': min(self.relaxation_param[1], self.relaxation_param[2]),
            'relaxation_yf': self.relaxation_param[3],
            'relaxation_R2': self.relaxation_R2
        }

        return fit_results_dict


class CurveSet:
    vd_dict: dict[tuple[str], bool] = {}
    cp_dict: dict[tuple[str], int] = {}

    def __init__(self, ident_labels: list[str], curve_dict: dict[tuple[str], Curve]) -> None:

        self.ident_labels = ident_labels
        self.curve_dict = curve_dict
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> Curve:
        if self._index < len(self.curve_dict):
            key = list(self.curve_dict.keys())[self._index]
            self._index += 1
            return self.curve_dict[key]
        else:
            raise StopIteration

    def __getitem__(self, key: tuple[str]) -> Curve:
        return self.curve_dict[key]

    def items(self) -> Iterable[tuple[tuple[str], Curve]]:
        return self.curve_dict.items()

    def pickle(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def keys(self) -> list[tuple[str]]:
        return list(self.curve_dict.keys())

    def reduce_data(self) -> None:
        for curve in self:
            curve.reduce_data()

    def remove_curve(self, key: tuple[str]) -> None:
        del self.curve_dict[key]

    def remove_unannotated(self) -> None:
        keys_to_remove = []
        for key, curve in self.items():
            if curve.contact_index == 0:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.remove_curve(key)

    def correct_all_virtual_defl(self) -> None:
        for curve in self:
            curve.correct_virtual_defl()

    def set_vd_by_annotations(self) -> None:
        for key, value in self.vd_dict.items():
            if value:
                self[key].correct_virtual_defl()

    def set_cp_by_annotations(self) -> None:
        for key, value in self.cp_dict.items():
            self[key].set_contact_index(value)

    def update_annotations(self, annotations_dict: dict[tuple[str], Any]) -> None:
        value = next(iter(annotations_dict.values()))
        if isinstance(value, bool):
            self.vd_dict = annotations_dict
        elif isinstance(value, int):
            self.cp_dict = annotations_dict

    def update_annotations_from_file(self, file: str) -> None:
        anno_tuple_dict = read_json_file(file)
        self.update_annotations(anno_tuple_dict)

    def adjust_to_contact(self) -> None:
        for curve in self:
            curve.adjust_to_contact()

    def prepare_fit_all(self) -> None:
        self.set_cp_by_annotations()
        self.remove_unannotated()
        self.set_vd_by_annotations()
        self.adjust_to_contact()

    def get_fit_all(self, probe_diameter: float, ind: float | list[float]) -> pd.DataFrame:
        self.prepare_fit_all()
        all_fit = []

        for key, curve in tqdm(self.items()):
            fit_results_dict = {label: key_value for label,
                                key_value in zip(self.ident_labels, key)}
            curve.get_fits_data(probe_diameter, ind)
            curve.get_approach_rates()
            fit_results_dict.update(curve.get_all_fits())
            fit_results = pd.DataFrame([fit_results_dict])
            all_fit.append(fit_results)

        combined_results = pd.concat(all_fit, ignore_index=True)

        return combined_results

    def export_figures(self) -> pdf.PdfMerger:
        merger = pdf.PdfMerger()
        for key, curve in tqdm(self.items()):
            curve.get_figs_data()
            title = ""
            for label, ident in zip(self.ident_labels, key):
                title += label + ident

            title = fr"$\text{{{title}}}$"
            contact_fit_fig = curve.get_contact_fig_pdf()
            contact_fit_fig.update_layout(title={'text': title})
            contact_fit_fig_pdf = io.BytesIO(
                contact_fit_fig.to_image(format='pdf'))
            merger.append(contact_fit_fig_pdf)

            dwell_relaxation_fit_fig = curve.get_dwell_relax_fig_pdf()
            dwell_relaxation_fit_fig.update_layout(title={'text': title})
            dwell_relaxation_fit_fig_pdf = io.BytesIO(
                dwell_relaxation_fit_fig.to_image(format='pdf'))
            merger.append(dwell_relaxation_fit_fig_pdf)

        return merger
