from dash import Dash, html

from ..data import downloader
from . import (
    ids,
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
                    html.Button(
                        children="Download Image Data",
                        id=ids.DOWNLOAD_IMAGEDATA,
                        n_clicks=0,
                        className="dash-button"
                    ),
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
