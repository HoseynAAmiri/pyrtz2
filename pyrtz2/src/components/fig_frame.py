from dash import Dash, html, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import json
from typing import Protocol
import pandas as pd

from . import ids, fig
from ..utils.utils import load, get_current_annotation


class Curve(Protocol):
    data: pd.DataFrame
    contact_index: int

    def set_contact_index(self, index: int = 0) -> None:
        ...

    def correct_virtual_defl(self) -> None:
        ...

    def get_approach(self) -> pd.DataFrame:
        ...


class Experiment(Protocol):
    def __getitem__(self, index: tuple[str, ...]) -> Curve:
        ...


class AFM(Protocol):
    experiment: Experiment


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.CONTACT_FIG, 'figure', allow_duplicate=True),
         Output(ids.FORCETIME_FIG, 'figure', allow_duplicate=True)],
        [Input(ids.CURVE_DROPDOWN, 'value'),
         Input(ids.ADJUST_CHECKLIST, 'value'),
         Input(ids.VD_ANNOTATIONS, 'data'),
         Input(ids.CP_ANNOTATIONS, 'data')],
        [State(ids.EXPERIMENT, 'data'),
         State(ids.CONTACT_FIG, 'figure'),
         State(ids.FORCETIME_FIG, 'figure'),
         ],
        prevent_initial_call=True
    )
    def show_data(curve_value, adjust, vd_data, cp_data, encoded_experiment, contact_fig, forcetime_fig):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        vd = get_current_annotation(curve_value, vd_data)
        if trigger_id == ids.CP_ANNOTATIONS and not vd:
            raise PreventUpdate

        if trigger_id == ids.ADJUST_CHECKLIST and not vd:
            raise PreventUpdate

        key = eval(curve_value)['key']
        cp = get_current_annotation(curve_value, cp_data)
        experiment: AFM = load(encoded_experiment)
        curve = experiment.experiment[key]
        curve.set_contact_index(cp)

        if curve.contact_index > 2 and vd:
            curve.correct_virtual_defl()

        approach = curve.get_approach()
        curve_data = curve.data.copy()
        contact_fig = fig.get_fig(contact_fig)
        contact_fig = fig.update_fig(
            contact_fig,
            approach['ind'],
            approach['f'],
            color='blue',
            mode='markers',
            hover=True,
        )

        forcetime_fig = fig.get_fig(forcetime_fig)
        forcetime_fig = fig.update_fig(
            forcetime_fig,
            curve_data['t'],
            curve_data['f'],
            mode='lines',
            color='red',
            hover=False,
        )

        if adjust:
            contact_fig = fig.adjust_to_contact(cp, contact_fig)
            forcetime_fig = fig.adjust_to_contact(cp, forcetime_fig)

        contact_fig = fig.update_contact_line(cp, contact_fig)

        if not trigger_id in (ids.VD_ANNOTATIONS, ids.CP_ANNOTATIONS):
            contact_fig.update_layout(
                xaxis={'autorange': True},
                yaxis={'autorange': True},
            )
            forcetime_fig.update_layout(
                xaxis={'autorange': True},
                yaxis={'autorange': True},
            )

        return contact_fig, forcetime_fig

    @app.callback(
        Output(ids.CP_ANNOTATIONS, 'data'),
        [Input(ids.CONTACT_FIG, 'clickData')],
        [State(ids.CURVE_DROPDOWN, 'value'),
         State(ids.CP_ANNOTATIONS, 'data')],
        prevent_initial_call=True
    )
    def handle_click(clickData, curve_value, cp_data):
        key = eval(curve_value)['key']
        cp_annotations = json.loads(cp_data)
        new_selected_index = clickData['points'][0]['pointIndex']
        cp_annotations[repr(key)] = new_selected_index
        return json.dumps(cp_annotations)

    return html.Div(
        className='figure',
        id=ids.FIG_HOLDER,
        style={
            'display': 'flex',
            'width': '100%',
        },
        children=[
            fig.render(id=ids.CONTACT_FIG,
                       title=r"$\text{Selected Contact Point: }$",
                       xaxis=r"$Z_{sensor} \text{ (m)}$"),
            fig.render(id=ids.FORCETIME_FIG,
                       title=r"$\text{Dwell and Relaxation}$",
                       xaxis=r"$Time \text{ (s)}$"),
        ],
    )