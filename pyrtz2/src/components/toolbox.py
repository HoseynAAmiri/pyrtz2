from dash import Dash, html

from ..data import downloader
from . import (
    annotator,
    fitter,
    contact_controls
)


def render(app: Dash) -> html.Div:

    return html.Div(
        className='toolbox',
        children=[
            annotator.render(app),
            html.Div(
                children=[
                    contact_controls.render(app),
                    fitter.render(app),
                ],
                style={
                    'display': 'flex',
                    'flex-direction': 'column',
                    'gap': '5px',
                    'align-items': 'start'
                },
            ),
            downloader.render(app),
        ],
        style={
            'width': '50%',
        },
    )
