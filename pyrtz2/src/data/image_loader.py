from dash import Dash, dcc
from dash.dependencies import Input, Output, State

import os
import glob

from ..components import ids
from ..utils.utils import (
    dump,
    extract_keys,
    group_values_by_keys
)


def render(app: Dash) -> dcc.Store:
    @app.callback(
        Output(ids.IMAGES, 'data', allow_duplicate=True),
        [Input(ids.EXPERIMENT, 'data')],
        [State(ids.EXPERIMENT_PATH, 'value'),
         State(ids.EXPERIMENT_LABELS, 'value')],
        prevent_initial_call=True
    )
    def store_images(_, experiment_path, labels):
        label_list = [label.strip() for label in labels.split(';')]

        images = {}
        all_images_dict = {}
        for tif_path in glob.glob(os.path.join(experiment_path, '*.tif')):
            curve_name = os.path.basename(tif_path).split('.')[0]
            curve_key = extract_keys(curve_name, label_list)
            all_images_dict[curve_key] = os.path.join(experiment_path, tif_path)
            images = group_values_by_keys(all_images_dict)

        return dump(images)

    return dcc.Store(id=ids.IMAGES)
