from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output, State
import json
import zipfile
import io

from pyrtz2.afm import AFM

from ..components import ids
from ..utils.utils import load, dump
from ..utils.processor import (
    process_experiment,
    process_images,
    process_indentation,
    process_pixel,
    get_pdf,
)


def render(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.DOWNLOAD, 'data', allow_duplicate=True),
        Input(ids.DOWNLOAD_ANNOTATIONS, 'n_clicks'),
        [State(ids.CP_ANNOTATIONS, 'data'),
         State(ids.VD_ANNOTATIONS, 'data'),
         State(ids.IM_ANNOTATIONS, 'data'),
         State(ids.LOG, 'children')],
        prevent_initial_call=True
    )
    def download_annotations(_, cp_data, vd_data, im_data, exp_output):
        exp_name = exp_output.split('\'')[1]

        annotations = {
            'cp_annotations.json': cp_data,
            'vd_annotations.json': vd_data,
            'im_annotations.json': im_data
        }

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zip_file:
            for filename, data in annotations.items():
                data_dict = json.loads(data)
                json_string = json.dumps(data_dict, indent=4)
                zip_file.writestr(f'{exp_name}_{filename}', json_string)
        buffer.seek(0)

        return dcc.send_bytes(buffer.getvalue(), f'{exp_name}_annotations.zip')

    @app.callback(
        [Output(ids.DOWNLOAD, 'data', allow_duplicate=True),
         Output(ids.EXPERIMENT, 'data', allow_duplicate=True),
         Output(ids.DOWNLOAD_FITS, 'children', allow_duplicate=True),
         Output(ids.INDENTATION, 'value', allow_duplicate=True)],
        Input(ids.DOWNLOAD_FITS, "n_clicks"),
        [State(ids.EXPERIMENT, 'data'),
         State(ids.IMAGES, 'data'),
         State(ids.CP_ANNOTATIONS, 'data'),
         State(ids.VD_ANNOTATIONS, 'data'),
         State(ids.IM_ANNOTATIONS, 'data'),
         State(ids.INDENTATION, 'value'),
         State(ids.PIXEL_SIZE, 'value'),
         State(ids.LOG, 'children')],
        prevent_initial_call=True
    )
    def download_fits_csv(_, encoded_experiment, encoded_images, cp_data, vd_data, im_data, indentation, pixel, exp_output):
        if indentation:
            experiment: AFM = load(encoded_experiment)
            indentation = process_indentation(indentation)
            experiment_processed, df_fits = process_experiment(
                experiment, cp_data, vd_data, indentation)
            exp_name = exp_output.split('\'')[1]

            images: dict = load(encoded_images)
            keys = experiment_processed.experiment.keys()
            pixel = process_pixel(pixel)
            df_images = process_images(images, keys, im_data, pixel)

            df = df_fits.join(df_images)
            return dcc.send_data_frame(df.to_csv, filename=f"{exp_name}_fits.csv"), dump(experiment_processed), no_update, no_update

        return no_update, no_update, no_update, "Unable to proceed without indentation!"

    @app.callback(
        [Output(ids.DOWNLOAD, 'data', allow_duplicate=True),
         Output(ids.DOWNLOAD_CURVES, 'children')],
        Input(ids.DOWNLOAD_CURVES, "n_clicks"),
        [State(ids.EXPERIMENT, 'data'),
         State(ids.LOG, 'children')],
        prevent_initial_call=True
    )
    def download_curves_pdf(_, encoded_experiment, exp_output):
        experiment_processed: AFM = load(encoded_experiment)
        exp_name = exp_output.split('\'')[1]
        pdf_merger = experiment_processed.experiment.export_figures()
        pdf_src = get_pdf(pdf_merger)

        return dcc.send_bytes(src=pdf_src.getvalue(), filename=f"{exp_name}_curves.pdf", base64=True), no_update
    '''
    @app.callback(
        [Output(ids.DOWNLOAD, 'data', allow_duplicate=True),
         Output(ids.DOWNLOAD_IMAGES, 'children')],
        Input(ids.DOWNLOAD_IMAGES, "n_clicks"),
        [State(ids.EXPERIMENT, 'data'),
         State(ids.IMAGES, 'data'),
         State(ids.LOG, 'children')],
        prevent_initial_call=True
    )
    def download_images_pdf(_, encoded_experiment, encoded_images, exp_output):
        experiment_processed = load(encoded_experiment)
        exp_name = exp_output.split('\'')[1]

        pdf_src = get_pdf(experiment_processed)
        images: dict = load(encoded_images)

        return dcc.send_bytes(src=pdf_src.getvalue(), filename=f"{exp_name}_curves.pdf", base64=True), no_update
    '''
    return html.Div(
        children=[
            dcc.Download(id=ids.DOWNLOAD),
            dcc.Loading(
                id=ids.DOWNLOAD_ANIMATION,
                type="dot",
                children=html.Div(
                    children=[
                        html.Button(
                            children="Download Fits",
                            id=ids.DOWNLOAD_FITS,
                            n_clicks=0,
                            className="dash-button"
                        ),
                        html.Button(
                            children="Download Curves",
                            id=ids.DOWNLOAD_CURVES,
                            n_clicks=0,
                            className="dash-button"
                        ),
                        html.Button(
                            children="Download Images",
                            id=ids.DOWNLOAD_IMAGES,
                            n_clicks=0,
                            className="dash-button"
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'gap': '5px',
                        'align-items': 'start',
                    }
                )
            ),
            html.Button(
                children="Download Experiment",
                id=ids.DOWNLOAD_EXPERIMENT,
                n_clicks=0,
                className="dash-button"
            ),
        ],
        style={
            'display': 'flex',
            'flex-direction': 'column',
            'width': '100%',
            'gap': '5px',
            'align-items': 'start',
        }
    )
