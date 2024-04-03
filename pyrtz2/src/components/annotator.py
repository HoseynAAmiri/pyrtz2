from dash import Dash, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import json

from . import ids
from ..utils.utils import load_json, update_annotations


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.CP_ANNOTATIONS, 'data', allow_duplicate=True),
         Output(ids.VD_ANNOTATIONS, 'data', allow_duplicate=True)],
        [Input(ids.UPLOAD_ANNOTATIONS, 'contents')],
        [State(ids.CP_ANNOTATIONS, 'data'),
         State(ids.VD_ANNOTATIONS, 'data')],
        prevent_initial_call=True
    )
    def load_annotations(encoded_contents, cp_data, vd_data):
        if not encoded_contents or not cp_data or not vd_data:
            raise PreventUpdate

        content_type, content_string = encoded_contents.split(',')
        loaded_annotations = load_json(content_string)
        first_value = next(iter(loaded_annotations.values()))
        if isinstance(first_value, bool):
            vd_annotations = json.loads(vd_data)
            vd_annotations = update_annotations(
                vd_annotations, loaded_annotations)
            return no_update, json.dumps(vd_annotations)
        elif isinstance(first_value, int):
            cp_annotations = json.loads(cp_data)
            cp_annotations = update_annotations(
                cp_annotations, loaded_annotations)
            return json.dumps(cp_annotations), no_update

    return html.Div(
        children=[
            dcc.Store(id=ids.CP_ANNOTATIONS),
            dcc.Store(id=ids.VD_ANNOTATIONS),
            dcc.Upload(
                className='drag-drop',
                children=html.Div(
                    ['Drag and Drop or ', html.A('Select Annotation File')]),
                id=ids.UPLOAD_ANNOTATIONS,
                multiple=False
            ),
        ],
    )
