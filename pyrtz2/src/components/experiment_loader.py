from dash import Dash, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import os

from ...afm import AFM
from ..components import ids
from ..utils.utils import make_json, save_afm


def render(app: Dash) -> dcc.Store:
    @app.callback(
        [Output(ids.EXPERIMENT_CACHE, 'data', allow_duplicate=True),
         Output(ids.CP_ANNOTATIONS, 'data', allow_duplicate=True),
         Output(ids.VD_ANNOTATIONS, 'data', allow_duplicate=True),
         Output(ids.IM_ANNOTATIONS, 'data', allow_duplicate=True)],
        [Input(ids.LOAD_EXPERIMENT, 'n_clicks')],
        [State(ids.EXPERIMENT_PATH, 'value'),
         State(ids.EXPERIMENT_LABELS, 'value'),
         State(ids.PROBE_DIAMETER, 'value')],
        prevent_initial_call=True
    )
    def store_experiment_info(n_clicks, experiment_path, labels, probe_diameter):
        if not experiment_path or not os.path.exists(experiment_path) or not os.path.isdir(experiment_path):
            raise PreventUpdate

        if not labels or not probe_diameter:
            raise PreventUpdate

        exp_name = os.path.basename(os.path.normpath(experiment_path))
        path = os.path.dirname(os.path.normpath(experiment_path))
        label_list = [label.strip() for label in labels.split(';')]
        experiment = AFM(path, exp_name, label_list, float(probe_diameter))
        experiment.experiment.reduce_data()
        cp_data = make_json(experiment.curve_keys, 0)
        vd_data = make_json(experiment.curve_keys, False)
        im_data = make_json(experiment.curve_keys, {'selection': 'exclude',
                                                    'clickData': []})

        cache_path = experiment_path + '/.cache'
        os.makedirs(cache_path, exist_ok=True)
        experiment_file_path = save_afm(cache_path, experiment, name='raw')

        experiment_temp = {
            'raw': experiment_file_path,
            'processed': cache_path,
        }

        return experiment_temp, cp_data, vd_data, im_data

    return dcc.Store(id=ids.EXPERIMENT_CACHE)
