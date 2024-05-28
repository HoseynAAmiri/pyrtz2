from dash import Dash, html, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import json
import os
from . import ids, fig
from ..utils.utils import get_current_annotation
from ..utils.processor import process_indentation


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.CONTACT_FIG, 'figure', allow_duplicate=True),
         Output(ids.DWELL_FIG, 'figure', allow_duplicate=True)],
        [Input(ids.CURVE_DROPDOWN, 'value'),
         Input(ids.ADJUST_CHECKLIST, 'value'),
         Input(ids.VD_ANNOTATIONS, 'data'),
         Input(ids.CP_ANNOTATIONS, 'data'),
         Input(ids.FIT_CHECKLIST, 'value'),
         Input(ids.INDENTATION, 'value')],
        [State(ids.PROBE_DIAMETER, 'value'),
         State(ids.EXPERIMENT_PATH, 'value'),
         State(ids.CONTACT_FIG, 'figure'),
         State(ids.DWELL_FIG, 'figure')],
        prevent_initial_call=True
    )
    def show_data(curve_value, adjust, vd_data, cp_data, fit, indentation, probe_diameter, experiment_path, contact_fig, dwell_fig):
        ctx = callback_context
        if not ctx.triggered or not curve_value or not vd_data or not cp_data:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == ids.INDENTATION and not fit:
            raise PreventUpdate

        vd = get_current_annotation(curve_value, vd_data)
        if trigger_id == ids.CP_ANNOTATIONS and not vd and not adjust:
            raise PreventUpdate

        indentation = process_indentation(indentation)
        if trigger_id == ids.FIT_CHECKLIST and indentation == 0.0:
            raise PreventUpdate

        if trigger_id == ids.FIT_CHECKLIST and fit and not adjust:
            adjust = True

        name = eval(curve_value)['name'] + ".ibw"
        file_name = os.path.join(experiment_path, name)
        cp = get_current_annotation(curve_value, cp_data)

        contact_fig, dwell_fig = fig.handle_figure(
            file_name, cp, vd, adjust, fit, float(probe_diameter), indentation)

        return contact_fig, dwell_fig

    @app.callback(
        Output(ids.CP_ANNOTATIONS, 'data'),
        Input(ids.CONTACT_FIG, 'clickData'),
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
        },
        children=[
            fig.render(id=ids.CONTACT_FIG,
                       title=r"$\text{Selected Contact Point: }$",
                       xaxis=r"$Indentation \text{ (m)}$"),
            fig.render(id=ids.DWELL_FIG,
                       title=r"$\text{Dwell and Relaxation}$",
                       xaxis=r"$Time \text{ (s)}$"),
        ],
    )
