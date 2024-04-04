import pickle
import re
import io
import base64
from PIL import Image
import json
import os
from datetime import datetime


def get_current_annotation(current_dropdown_value, data: str):
    key = eval(current_dropdown_value)['key']
    annotations = json.loads(data)
    annotation = annotations[repr(key)]
    return annotation


def update_annotations(original_annotations: dict, loaded_annotations: dict) -> dict:
    for key, value in loaded_annotations.items():
        if key in original_annotations:
            original_annotations[key] = value
    return original_annotations


def make_folder(load_dir: str, name: str = "") -> str:
    if not name:
        time = datetime.now()
        name = time.strftime('%Y-%m-%d_%H-%M-%S') + '/'
    save_dir = load_dir + '/' + name
    os.mkdir(save_dir)
    return save_dir


def make_json(keys, value):
    dict = {}
    for key in keys:
        dict[repr(key)] = value
    return json.dumps(dict)


def load_json(content_string):
    decoded = base64.b64decode(content_string)
    loaded_json = json.loads(decoded.decode('utf-8'))
    return loaded_json


def load(encoded_data: str):
    decoded_data = bytes.fromhex(encoded_data)
    loaded_data = pickle.loads(decoded_data)
    return loaded_data


def dump(data):
    serialized_data = pickle.dumps(data)
    encoded_data = serialized_data.hex()
    return encoded_data


def extract_keys(name: str, labels: list[str]) -> tuple[str, ...]:
    regex_str = ""
    for l in labels:
        regex_str = regex_str + l + f'(?P<{l}>.*)'
    regex = re.compile(regex_str)
    matched = regex.match(name)
    if matched:
        key = tuple([matched.group(label) for label in labels])
        return key
    else:
        return ()


def group_values_by_keys(original_dict: dict) -> dict:
    if len(next(iter(original_dict))) == 1:
        return original_dict

    new_dict = {}
    for key, value in original_dict.items():
        new_key = key[:-1]
        if new_key not in new_dict:
            new_dict[new_key] = [value]
        else:
            new_dict[new_key].append(value)
    return new_dict


def tif_to_base64(tif_path):
    img = Image.open(tif_path)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")  # can be changeg to "JPEG"
    return base64.b64encode(buffer.getvalue()).decode()
