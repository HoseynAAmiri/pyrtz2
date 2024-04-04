from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State


from . import ids


def render(app: Dash) -> html.Div:

    return html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Checklist(
                        id=ids.FIT_CHECKLIST,
                        options=[
                            {'label': '  Show Fits', 'value': True}],
                        style={
                            'width': "25%",
                        },
                    ),
                    dcc.Input(
                        placeholder="Enter max indentation or interval...",
                        id=ids.INDENTATION,
                        style={
                            'width': "75%",
                        },
                    )
                ],
                style={
                    'display': 'flex',
                    'width': '100%',
                }
            ),
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
                            children="Download Experiment",
                            id=ids.DOWNLOAD_EXPERIMENT,
                            n_clicks=0,
                            className="dash-button"
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'gap': '5px',
                    },
                )
            )
        ],
        style={
            'display': 'flex',
            'flex-direction': 'column',
            'width': '100%',
            'gap': '5px',
            'align-items': 'start',
        }
    )
