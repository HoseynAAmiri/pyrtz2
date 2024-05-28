from dash import Dash, dcc, html, callback_context, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from typing import Protocol
import json

from . import ids, fig
from ..utils.utils import load, get_current_annotation


class Curve(Protocol):
    def detect_contact(self) -> int:
        ...


class CurveSet(Protocol):
    def __getitem__(self, key: tuple[str]) -> Curve:
        ...


class AFM(Protocol):
    experiment: CurveSet


def render(app: Dash) -> html.Div:

    @app.callback(
        [Output(ids.VD_CHECKLIST, 'value', allow_duplicate=True),
         Output(ids.VD_ANNOTATIONS, 'data')],
        [Input(ids.CURVE_DROPDOWN, 'value'),
         Input(ids.VD_ANNOTATIONS, 'data'),
         Input(ids.VD_CHECKLIST, 'value')],
        prevent_initial_call=True
    )
    def show_vd_checklist(curve_value, vd_data, vd_checklist):
        ctx = callback_context
        if not ctx.triggered or not curve_value or not vd_data:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        key = eval(curve_value)['key']
        vd_annotations = json.loads(vd_data)
        vd = vd_annotations[repr(key)]
        if trigger_id in (ids.CURVE_DROPDOWN, ids.VD_ANNOTATIONS):
            if vd:
                return [vd], no_update
            else:
                return [], no_update
        elif trigger_id == ids.VD_CHECKLIST:
            if vd_checklist:
                vd_annotations[repr(key)] = True
            else:
                vd_annotations[repr(key)] = False
            return no_update, json.dumps(vd_annotations)

    @app.callback(
        Output(ids.CONTACT_FIG, 'figure', allow_duplicate=True),
        [Input(ids.CP_ANNOTATIONS, 'data')],
        [State(ids.ADJUST_CHECKLIST, 'value'),
         State(ids.CURVE_DROPDOWN, 'value'),
         State(ids.CONTACT_FIG, 'figure'),
         State(ids.VD_CHECKLIST, 'value')],
        prevent_initial_call=True
    )
    def set_contact(cp_data, adjust, curve_value, contact_fig_dict, vd):
        if not curve_value or vd or adjust:
            raise PreventUpdate

        cp = get_current_annotation(curve_value, cp_data)
        contact_fig = fig.update_contact_line(cp, contact_fig_dict)
        return contact_fig

    @app.callback(
        [Output(ids.CP_ANNOTATIONS, 'data', allow_duplicate=True),
         Output(ids.DETECT_CONTACT, 'children')],
        [Input(ids.RESET_CONTACT, 'n_clicks'),
         Input(ids.DETECT_CONTACT, 'n_clicks')],
        [State(ids.CURVE_DROPDOWN, 'value'),
         State(ids.CP_ANNOTATIONS, 'data'),
         State(ids.EXPERIMENT, 'data')],
        prevent_initial_call=True
    )
    def update_contact(reset, detect, curve_value, cp_data, encoded_experiment):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        key = eval(curve_value)['key']
        cp_annotations = json.loads(cp_data)
        if trigger_id == ids.RESET_CONTACT:
            cp_annotations[repr(key)] = 0
        elif trigger_id == ids.DETECT_CONTACT:
            afm: AFM = load(encoded_experiment)
            cp_annotations[repr(
                key)] = afm.experiment[key].detect_contact()

        return json.dumps(cp_annotations), no_update

    return html.Div(
        children=[
            dcc.Checklist(id=ids.VD_CHECKLIST,
                          options=[{'label': '  Correct Virtual Deflection', 'value': True}]),
            dcc.Checklist(id=ids.ADJUST_CHECKLIST,
                          options=[{'label': '  Adjust to Contact', 'value': True}]),
            dcc.Loading(
                id=ids.LOADCONTACT_ANIMATION,
                type="dot",
                children=html.Div(
                    children=[
                        html.Button(
                            children="Detect Contact",
                            id=ids.DETECT_CONTACT,
                            n_clicks=0,
                            className="dash-button"
                        ),
                        html.Button(
                            children="Reset Contact",
                            id=ids.RESET_CONTACT,
                            n_clicks=0,
                            className="dash-button"
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'gap': '5px',
                    },
                ),
            )
        ],
        style={
            'display': 'flex',
            'flex-direction': 'column',
            'gap': '5px',
            'align-items': 'start'
        }
    )
