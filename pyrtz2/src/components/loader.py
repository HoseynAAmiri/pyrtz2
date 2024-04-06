from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import os

from . import ids
from ..data import (
    experiment_loader,
    image_loader
)


def render(app: Dash) -> html.Div:

    @app.callback(
        Output(ids.LOG, 'children'),
        Input(ids.LOAD_EXPERIMENT, 'n_clicks'),
        [State(ids.EXPERIMENT_PATH, 'value'),
         State(ids.EXPERIMENT_LABELS, 'value'),
         State(ids.PROBE_DIAMETER, 'value')],
        prevent_initial_call=True
    )
    def update_output(_, experiment_path, labels, probe_diameter):
        if not experiment_path or not os.path.exists(experiment_path) or not os.path.isdir(experiment_path):
            return html.Div("Invalid directory path.", id=ids.LOG)

        if not labels or not probe_diameter:
            return html.Div("Labels or probe diameter cannot be empty.", id=ids.LOG)

        exp_name = os.path.basename(os.path.normpath(experiment_path))
        return html.Div(f"Experiment '{exp_name}' loading.", id=ids.LOG)

    @app.callback(
        Output(ids.EXPERIMENT_PATH, 'style'),
        Input(ids.EXPERIMENT_PATH, 'value'),
        State(ids.EXPERIMENT_PATH, 'style')
    )
    def shrink_output(value, current_style):
        if not value:
            current_style['direction'] = 'ltr'
            return current_style
        current_style['direction'] = 'rtl'
        return current_style

    return html.Div(
        children=[
            html.H5("Load Experiment"),
            html.Div(
                children=[
                    dcc.Input(
                        id=ids.EXPERIMENT_PATH,
                        type='text',
                        placeholder='Enter experiment folder path...',
                        style={
                            'flex': '1',
                            'margin': '2px'}
                    ),
                    dcc.Input(
                        id=ids.EXPERIMENT_LABELS,
                        type='text',
                        placeholder='Enter labels separated by semicolons...',
                        style={
                            'flex': '1',
                            'margin': '2px'}
                    ),
                    dcc.Input(
                        id=ids.PROBE_DIAMETER,
                        type='text',
                        placeholder='Enter probe diameter...',
                        style={
                            'margin': '2px',
                            'width': '20%'}
                    ),
                    html.Button(
                        children="Load",
                        id=ids.LOAD_EXPERIMENT,
                        n_clicks=0,
                        style={
                            'margin': '2px'
                        },
                        className="dash-button"
                    ),
                ],
                style={
                    'display': 'flex',
                    'width': '100%',
                }
            ),
            experiment_loader.render(app),
            image_loader.render(app),
        ],
        className='experiment-loader',
    )
