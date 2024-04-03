from dash import Dash, dcc, html
from . import (
    ids,
    loader,
    curve_dropdown,
    image_frame,
    toolbox,
    fig_frame,
)


def create_layout(app: Dash) -> html.Div:
    return html.Div(
        className="app-div",
        children=[
            html.H1(app.title),
            html.Hr(),
            dcc.Loading(
                id=ids.LOAD_ANIMATION,
                # There are different types like 'graph', 'cube', 'circle', 'dot', and 'default'
                type="default",
                children=html.Div(
                    className="file-reader",
                    children=[
                        loader.render(app),
                    ],
                ),
            ),
            html.Hr(),
            html.Div(
                className='dashboard',
                children=[
                    html.Div(
                        children=[
                            curve_dropdown.render(app),
                            dcc.Loading(
                                id=ids.LOADIMAGE_ANIMATION,
                                type="default",
                                children=[image_frame.render(app)]
                            ),
                        ],
                        style={
                            'width': '50%',
                        },
                    ),
                    toolbox.render(app),
                ],
                style={
                    'display': 'flex',
                    'width': '100%',
                },
            ),
            html.Hr(),
            fig_frame.render(app)
        ],
    )
