from dash import Dash, dcc, html, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from typing import Protocol

from . import ids
from ..utils.utils import load_afm


class AFM(Protocol):
    curve_keys: list[tuple[str, ...]]


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.CURVE_DROPDOWN, 'options'),
         Output(ids.CURVE_DROPDOWN, 'value', allow_duplicate=True),
         Output(ids.LOG, 'children', allow_duplicate=True)],
        [Input(ids.EXPERIMENT_CACHE, 'data')],
        [State(ids.EXPERIMENT_LABELS, 'value'),
         State(ids.EXPERIMENT_PATH, 'value')],
        prevent_initial_call=True
    )
    def update_curve_dropdown(experiment_temp, labels, experiment_path):
        experiment = load_afm(experiment_temp['raw'])
        keys = experiment.curve_keys
        label_list = [label.strip() for label in labels.split(';')]

        options = []
        for idx, key in enumerate(keys):
            name = ""
            dropdown_elements = []
            for label, k in zip(label_list, key):
                name += label + k
                dropdown_elements.append(html.Span(label))  # Normal text
                dropdown_elements.append(html.Strong(k))    # Bold text

            dropdown_elements.append(html.Span(' ('))
            dropdown_elements.append(html.Strong(str(idx+1)))
            dropdown_elements.append(
                html.Span('/' + str(len(keys)) + ')'))  # Index text

            dropdown_label = html.Div(dropdown_elements, style={
                                      'whiteSpace': 'pre-wrap'})

            dropdown_value = repr(
                {
                    'index': idx,
                    'key': key,
                    'name': name
                }
            )
            options.append({'label': dropdown_label, 'value': dropdown_value})

        exp_name = experiment_path.split('\\')[-1]
        return options, options[0]['value'], f"Experiment '{exp_name}' loaded."

    @app.callback(
        Output(ids.CURVE_DROPDOWN, 'value'),
        [Input(ids.BUTTON_BACK, 'n_clicks'),
         Input(ids.BUTTON_FORWARD, 'n_clicks')],
        [State(ids.CURVE_DROPDOWN, 'value'),
         State(ids.CURVE_DROPDOWN, 'options')],
        prevent_initial_call=True
    )
    def move_buttons(up_clicks, down_clicks, current_value, options):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        current_index = next((i for i, option in enumerate(
            options) if option['value'] == current_value), None)

        if current_index is None:
            raise PreventUpdate

        if trigger_id == ids.BUTTON_FORWARD:
            new_index = min(len(options) - 1, current_index + 1)
        elif trigger_id == ids.BUTTON_BACK:
            new_index = max(0, current_index - 1)
        else:
            raise PreventUpdate

        if current_index == new_index:
            raise PreventUpdate

        return options[new_index]['value']

    return html.Div(
        children=[
            html.Button(
                children='<',
                id=ids.BUTTON_BACK,
                n_clicks=0,
                className="dash-button"
            ),
            dcc.Dropdown(
                id=ids.CURVE_DROPDOWN,
                options=[],
                clearable=False,
                style={
                    'width': '99%',
                    'justifyContent': 'space-around',
                    'margin': '0 auto',
                },
            ),
            html.Button(
                children='>',
                id=ids.BUTTON_FORWARD,
                n_clicks=0,
                className="dash-button"
            ),
        ],
        style={
            'display': 'flex',
            'width': '95%',
            'justifyContent': 'space-around',
            'margin': '0 auto',
        },
    )
