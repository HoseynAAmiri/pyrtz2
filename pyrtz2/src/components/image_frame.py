import json

from dash import Dash, html, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from . import ids, image
from ..utils.utils import load, get_current_annotation, parse_path


def render(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.IMAGE_HOLDER, 'figure', allow_duplicate=True),
        [Input(ids.CURVE_DROPDOWN, 'value'),
         Input(ids.IM_ANNOTATIONS, 'data')],
        State(ids.IMAGES, 'data'),
        prevent_initial_call=True
    )
    def update_image_frame(curve_value, im_data, encoded_images):
        if not curve_value:
            raise PreventUpdate

        key = eval(curve_value)['key']
        new_key = key[:-1] if len(key) > 1 else key

        images: dict = load(encoded_images)

        if new_key in images:
            im = get_current_annotation(curve_value, im_data)
            # THIS ONLY SHOWS THE FIRST IMAGE IF THERE IS ONE
            image_path = images[new_key][0]
            img = image.handle_image(image_path, im)
            return img
        else:
            return image.make()

    @app.callback(
        [Output(ids.IM_ANNOTATIONS, 'data', allow_duplicate=True),
         Output(ids.IMAGE_HOLDER, 'clickData')],
        Input(ids.IMAGE_HOLDER, 'clickData'),
        [State(ids.CURVE_DROPDOWN, 'value'),
         State(ids.IM_ANNOTATIONS, 'data'),
         State(ids.IMAGE_HOLDER, 'figure')],
        prevent_initial_call=True
    )
    def handle_click(clickData, curve_value, im_data, fig):
        key = eval(curve_value)['key']
        im_annotations = json.loads(im_data)
        selection = im_annotations[repr(key)]['selection']
        if selection == 'auto':
            x, y = clickData['points'][0]['y'], clickData['points'][0]['x']
            im_annotations[repr(key)]['clickData'] = [[x, y]]
            return json.dumps(im_annotations), None

        if selection == 'manual' and 'shapes' in fig['layout']:
            shapes = fig['layout']['shapes'][0]
            im_annotations[repr(key)]['clickData'] = parse_path(shapes['path'])
            fig['layout']['shapes'] = []
            return json.dumps(im_annotations), None

        return no_update, None

    return html.Div(
        className='image',
        children=[
            image.render(ids.IMAGE_HOLDER)
        ],
        style={
            'height': '280px',
            'width': 'auto',
        },
    )
