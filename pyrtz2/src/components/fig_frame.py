from dash import Dash, html, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import json
import os
from . import ids, fig
from ..utils.utils import get_current_annotation
from ...asylum import load_ibw


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.CONTACT_FIG, 'figure', allow_duplicate=True),
         Output(ids.FORCETIME_FIG, 'figure', allow_duplicate=True)],
        [Input(ids.CURVE_DROPDOWN, 'value'),
         Input(ids.ADJUST_CHECKLIST, 'value'),
         Input(ids.VD_ANNOTATIONS, 'data'),
         Input(ids.CP_ANNOTATIONS, 'data')],
        [State(ids.EXPERIMENT_PATH, 'value'),
         State(ids.CONTACT_FIG, 'figure'),
         State(ids.FORCETIME_FIG, 'figure'),
         ],
        prevent_initial_call=True
    )
    def show_data(curve_value, adjust, vd_data, cp_data, experiment_path, contact_fig, forcetime_fig):
        ctx = callback_context
        if not ctx.triggered or not curve_value or not vd_data or not cp_data:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        vd = get_current_annotation(curve_value, vd_data)
        if trigger_id == ids.CP_ANNOTATIONS and not vd and not adjust:
            raise PreventUpdate
        print('raft too')
        name = eval(curve_value)['name'] + ".ibw"
        file_name = os.path.join(experiment_path, name)
        curve = load_ibw(file_name)
        curve.reduce_data()
        cp = get_current_annotation(curve_value, cp_data)
        curve.set_contact_index(cp)
        curve.get_figs_data(vd=vd, adjust=adjust)
        contact_fig = curve.get_contact_fig_plot()
        forcetime_fig = curve.get_dwell_relax_fig_plot()
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
                       xaxis=r"$Indentation \text{ (m)}$"),
            fig.render(id=ids.FORCETIME_FIG,
                       title=r"$\text{Dwell and Relaxation}$",
                       xaxis=r"$Time \text{ (s)}$"),
        ],
    )
