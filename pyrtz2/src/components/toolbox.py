from dash import Dash, html

from . import (
    ids,
    annotator,
    contact_controls,
    image_controls,
    fitter,
    downloader
)


def render(app: Dash) -> html.Div:

    return html.Div(
        className='toolbox',
        children=[
            annotator.render(app),
            html.Div(
                children=[
                    contact_controls.render(app),
                    image_controls.render(app),
                    html.Button(
                        children="Download Annotations",
                        id=ids.DOWNLOAD_ANNOTATIONS,
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
            'display': 'flex',
            'flex-direction': 'column',
            'gap': '5px',
            'width': '50%',
        },
    )
