from dash import Dash, html
from dash.dependencies import Input, Output

from . import ids
from ..utils.utils import load


def render(app: Dash) -> html.Div:

    @app.callback(
        Output(ids.IMAGE_FRAME, 'src'),
        [Input(ids.CURVE_DROPDOWN, 'value'),
         Input(ids.IMAGES, 'data')],
        prevent_initial_call=True
    )
    def update_image_frame(curve_value, encoded_images):
        if not encoded_images:
            return ''

        images: dict = load(encoded_images)

        key = eval(curve_value)['key']
        new_key = key[:-1] if len(key) > 1 else key

        # THIS ONLY SHOWS THE FIRST IMAGE IF THERE IS ONE
        if images.get(new_key):
            image = images[new_key][0]
            image_src = f'data:image/png;base64,{image}'
            return image_src
        else:
            return ''

    return html.Div(
        className='image',
        children=[
            html.Img(
                id=ids.IMAGE_FRAME,
                style={'display': 'flex',
                       'margin': '10px auto',
                       'height': '250px'
                       }
            )
        ],
    )
