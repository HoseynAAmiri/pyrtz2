from dash import Dash, dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State


from . import ids
from ..utils.processor import process_indentation


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.ADJUST_CHECKLIST, 'value'),
         Output(ids.FIT_CHECKLIST, 'value'),
         Output(ids.INDENTATION, 'value')],
        [Input(ids.ADJUST_CHECKLIST, 'value'),
         Input(ids.FIT_CHECKLIST, 'value')],
        State(ids.INDENTATION, 'value'),
        prevent_initial_call=True
    )
    def link_adjust_fit(adjust, fit, indentation):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        indentation = process_indentation(indentation)

        if trigger_id == ids.FIT_CHECKLIST and indentation == 0.0:
            return no_update, [], "Unable to proceed without indentation!"

        if trigger_id == ids.ADJUST_CHECKLIST and fit:
            fit = []

        if trigger_id == ids.ADJUST_CHECKLIST and not fit:
            return no_update

        if trigger_id == ids.FIT_CHECKLIST and not adjust:
            adjust = [True]

        return adjust, fit, no_update

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
        ],
        style={
            'display': 'flex',
            'flex-direction': 'column',
            'width': '100%',
            'gap': '5px',
            'align-items': 'start',
        }
    )
