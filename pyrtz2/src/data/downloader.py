from dash import Dash, dcc, no_update
from dash.dependencies import Input, Output, State
import json

from ..components import ids
from ..utils.utils import load, dump
from .processor import process_experiment, process_indentation, get_pdf


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
         Output(ids.EXPERIMENT, 'data', allow_duplicate=True),
         Output(ids.DOWNLOAD_FITS, 'children', allow_duplicate=True),
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
            indentation = process_indentation(indentation)
            experiment_processed, df = process_experiment(
                experiment, cp_data, vd_data, indentation)
            exp_name = exp_output.split('\'')[1]
            return dcc.send_data_frame(df.to_csv, filename=f"{exp_name}_fits.csv"), dump(experiment_processed), no_update, no_update

        return no_update, no_update, no_update, "Unable to proceed without indentation!"

    @app.callback(
        [Output(ids.DOWNLOAD, 'data', allow_duplicate=True),
         Output(ids.DOWNLOAD_CURVES, 'children')],
        [Input(ids.DOWNLOAD_CURVES, "n_clicks")],
        [State(ids.EXPERIMENT, 'data'),
         State(ids.LOAD_OUTPUT, 'children')],
        prevent_initial_call=True
    )
    def download_curves_pdf(_, encoded_experiment, exp_output):
        experiment_processed = load(encoded_experiment)
        exp_name = exp_output.split('\'')[1]
        pdf_src = get_pdf(experiment_processed)

        return dcc.send_bytes(src=pdf_src.getvalue(), filename=f"{exp_name}_curves.pdf", base64=True), no_update

    return dcc.Download(id=ids.DOWNLOAD)
