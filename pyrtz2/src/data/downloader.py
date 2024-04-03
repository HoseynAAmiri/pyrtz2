from dash import Dash, dcc, no_update
from dash.dependencies import Input, Output, State
import json

from ..components import ids
from ..utils.utils import load, dump
from .processor import process_experiment, get_all_fit


def render(app: Dash) -> dcc.Download:
    @app.callback(
        Output(ids.DOWNLOAD, 'data', allow_duplicate=True),
        [Input(ids.DOWNLOAD_ANNOTATIONS, 'n_clicks')],
        [State(ids.CP_ANNOTATIONS, 'data'),
         State(ids.LOAD_OUTPUT, 'children')],
        prevent_initial_call=True
    )
    def download_cp(_, cp_data, exp_output):
        exp_name = exp_output.split('\'')[1]
        data_dict = json.loads(cp_data)

        json_string = json.dumps(data_dict, indent=4)
        return {'content': json_string, 'filename': f'{exp_name}_cp_annotations.json', 'type': 'text/json'}

    @app.callback(
        Output(ids.DOWNLOAD, 'data'),
        [Input(ids.DOWNLOAD_ANNOTATIONS, 'n_clicks')],
        [State(ids.VD_ANNOTATIONS, 'data'),
         State(ids.LOAD_OUTPUT, 'children')],
        prevent_initial_call=True
    )
    def download_vd(_, vd_data, exp_output):
        exp_name = exp_output.split('\'')[1]
        data_dict = json.loads(vd_data)

        json_string = json.dumps(data_dict, indent=4)
        return {'content': json_string, 'filename': f'{exp_name}_vd_annotations.json', 'type': 'text/json'}

    @app.callback(
        [Output(ids.DOWNLOAD, 'data', allow_duplicate=True),
         Output(ids.EXPERIMENT_PROCESSED, 'data'),
         Output(ids.DOWNLOAD_FITS, 'children'),
         Output(ids.INDENTATION, 'value')],
        [Input(ids.DOWNLOAD_FITS, "n_clicks")],
        [State(ids.EXPERIMENT, 'data'),
         State(ids.CP_ANNOTATIONS, 'data'),
         State(ids.VD_ANNOTATIONS, 'data'),
         State(ids.INDENTATION, 'value'),
         State(ids.LOAD_OUTPUT, 'children')],
        prevent_initial_call=True
    )
    def download_fits_csv(_, encoded_experiment, cp_data, vd_data, indentation, exp_output):
        if indentation:
            experiment = load(encoded_experiment)
            experiment_processed = process_experiment(
                experiment, cp_data, vd_data)
            indentation = float(indentation)
            exp_name = exp_output.split('\'')[1]
            df = get_all_fit(experiment_processed, indentation=indentation)
            return dcc.send_data_frame(df.to_csv, filename=f"{exp_name}_all_fits.csv"), dump(experiment_processed), no_update, no_update

        return no_update, no_update, no_update, "Unable to proceed without indentation!"

    return dcc.Download(id=ids.DOWNLOAD)
