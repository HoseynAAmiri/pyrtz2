from dash import Dash, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import os

from afm import AFM
from ..components import ids
from ..utils.utils import dump, make_json


def render(app: Dash) -> dcc.Store:
    @app.callback(
        [Output(ids.EXPERIMENT, 'data', allow_duplicate=True),
         Output(ids.EXPERIMENT_PROCESSED, 'data', allow_duplicate=True),
         Output(ids.CP_ANNOTATIONS, 'data', allow_duplicate=True),
         Output(ids.VD_ANNOTATIONS, 'data', allow_duplicate=True)],
        [Input(ids.LOAD_EXPERIMENT, 'n_clicks')],
        [State(ids.EXPERIMENT_PATH, 'value'),
         State(ids.EXPERIMENT_LABELS, 'value'),
         State(ids.PROBE_DIAMETER, 'value')],
        prevent_initial_call=True
    )
    def store_experiment_info(_, experiment_path, labels, probe_diameter):
        if not experiment_path or not os.path.exists(experiment_path) or not os.path.isdir(experiment_path):
            raise PreventUpdate

        if not labels or not probe_diameter:
            raise PreventUpdate

        exp_name = os.path.basename(os.path.normpath(experiment_path))
        path = os.path.dirname(os.path.normpath(experiment_path))
        label_list = [label.strip() for label in labels.split(';')]
        experiment = AFM(path, exp_name, label_list, float(probe_diameter))

        cp_data = make_json(experiment.curve_keys, 0)
        vd_data = make_json(experiment.curve_keys, False)
        return dump(experiment), dump(experiment), cp_data, vd_data

    return dcc.Store(id=ids.EXPERIMENT)
