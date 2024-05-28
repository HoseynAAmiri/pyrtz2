from dash import Dash, dcc, html, no_update, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import json

from . import ids


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.IM_ANNOTATIONS, 'data', allow_duplicate=True),
         Output(ids.CLIP_LIMIT, 'value'),
         Output(ids.SQUARE_VALUE, 'value'),
         Output(ids.IM_DROPDOWN, 'value')],
        [Input(ids.CURVE_DROPDOWN, 'value'),
         Input(ids.CLIP_LIMIT, 'value'),
         Input(ids.SQUARE_VALUE, 'value'),
         Input(ids.IM_DROPDOWN, 'value')],
        State(ids.IM_ANNOTATIONS, 'data'),
        prevent_initial_call=True
    )
    def update_im(curve_value, clipLimit, sq, im_value, im_data):
        if not curve_value:
            raise PreventUpdate

        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        key = eval(curve_value)['key']
        im_annotations = json.loads(im_data)
        im = im_annotations[repr(key)]
        if trigger_id == ids.CURVE_DROPDOWN:
            selection = im['selection']
            if selection == 'auto':
                clipLimit = im['contrast']
                sq = im['size']
                return no_update, clipLimit, sq, selection
            else:
                return no_update, no_update, no_update, selection

        im['selection'] = im_value
        im['clickData'] = []
        if im_value == 'auto':
            im['contrast'] = clipLimit
            im['size'] = sq
        if im_value == 'manual' and im.get('contrast') and im.get('size'):
            del im['contrast']
            del im['size']
        im_annotations[repr(key)] = im
        return json.dumps(im_annotations), no_update, no_update, im_value

    @app.callback(
        [Output(ids.CLIP_LIMIT, 'disabled'),
         Output(ids.SQUARE_VALUE, 'disabled')],
        Input(ids.IM_DROPDOWN, 'value')
    )
    def update_input_state(selected_value):
        if selected_value == 'auto':
            return False, False
        else:
            return True, True

    return html.Div(
        children=[
            html.Div(children="Contrast: "),
            dcc.Input(
                id=ids.CLIP_LIMIT,
                type='number',
                value=1.5,
                step=0.1,
                min=0.1,
                style={'width': '25%'}
            ),
            html.Div(children="Size: "),
            dcc.Input(
                id=ids.SQUARE_VALUE,
                type='number',
                value=5,
                step=1,
                min=2,
                style={'width': '25%'}
            ),
            dcc.Input(
                id=ids.PIXEL_SIZE,
                type='text',
                placeholder='Pixel size...',
                style={'width': '50%'}
            ),
            dcc.Dropdown(
                id=ids.IM_DROPDOWN,
                options=[
                    {'label': 'Exclude', 'value': 'exclude'},
                    {'label': 'Auto', 'value': 'auto'},
                    {'label': 'Manual', 'value': 'manual'}
                ],
                value='exclude',
                clearable=False,
                style={'width': '99%'}
            ),
        ],
        style={
            'display': 'flex',
            'width': '100%',
            'gap': '5px',
            'align-items': 'start',
        }
    )
