import numpy as np
import numpy.typing as npt
from typing import Callable

import cv2
from skimage import filters
from skimage.morphology import closing, square, remove_small_objects
from skimage.feature import canny
from skimage.measure import label as _label
from scipy import ndimage as ndi

import plotly.graph_objects as go
from dash import dcc


def render(id: str) -> dcc.Graph:
    return dcc.Graph(
        id=id,
        figure=make(),
        style={
            'display': 'flex',
            'height': '100%',
        }
    )


def plot(img: go.Figure, image_array: npt.NDArray[np.uint8]) -> go.Figure:
    img.add_trace(go.Image(z=image_array))
    return img


def make() -> go.Figure:
    img = go.Figure()
    img.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return img


def read(image_path: str) -> npt.NDArray[np.uint8]:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return image.astype(np.uint8)


def _contrast(gray: npt.NDArray[np.uint8], clipLimit=2.0) -> npt.NDArray[np.uint8]:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)


def _filter(gray: npt.NDArray[np.uint8], func: Callable) -> npt.NDArray[np.uint8]:
    thr = func(gray)
    thr = 127 - abs(thr - 127)
    mask = (gray < thr) * 255
    mask = mask.astype(np.uint8)
    return mask


def label(image: npt.NDArray[np.uint8], clipLimit=2.0, sq: int = 5) -> npt.NDArray[np.uint8]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray = _contrast(gray, clipLimit)
    closed = closing(gray, square(sq))
    fg_gray = (closed - gray)
    filtered = _filter(fg_gray, filters.threshold_triangle)
    edges = canny(filtered)
    filled = ndi.binary_fill_holes(edges)
    cleaned = remove_small_objects(filled, 10)
    labels = np.array(_label(cleaned), dtype=np.uint8)
    return labels


def _contours(mask: npt.NDArray[np.uint8], count: int = 100) -> list[list[list[int]]]:
    cell_contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(
        cell_contours, key=cv2.contourArea, reverse=True)[:count]
    areas = np.array([cv2.contourArea(contour)
                      for contour in sorted_contours])
    valid_indices = np.where(0.2 < (areas / np.max(areas)))[0]
    cell_contours = [sorted_contours[i] for i in valid_indices]
    return cell_contours


def _clean(image: npt.NDArray[np.uint8], cell_contour: list[list[int]]) -> list[list[list[int]]]:
    cell_mask = np.zeros_like(image)
    cell_hull = cv2.convexHull(np.array(cell_contour))
    cv2.fillPoly(cell_mask, [cell_hull], (255, 255, 255))

    # , shrink: int = 0
    # if shrink > 0:
    #     kernel_size = 5
    #     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #     cell_mask = cv2.erode(
    #         cell_mask, kernel, iterations=shrink).astype(np.uint8)

    mask = cell_mask.astype(np.uint8)
    cell_contours = _contours(mask)
    return cell_contours


def _mask(image: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8], alpha: float = 0.5) -> npt.NDArray[np.uint8]:
    masked_pixels = np.where(mask == 255)
    image[masked_pixels] = (alpha * mask[masked_pixels] +
                            (1 - alpha) * image[masked_pixels])
    return image


def get_contours(labels: npt.NDArray[np.uint8], clean: bool = True) -> list[list[list[int]]]:
    cell_contours = _contours(labels)
    if len(cell_contours) == 1 and clean:
        cell_contours = _clean(labels, cell_contours[0])
    return cell_contours


def add_contours(image: npt.NDArray[np.uint8], cell_contours: list[list[list[int]]]) -> npt.NDArray[np.uint8]:
    if len(cell_contours) == 1:
        cell_mask = np.zeros_like(image)
        cv2.fillPoly(cell_mask, [np.array(cell_contours[0])], (255, 0, 0))
        image = _mask(image, cell_mask)

    for contour in cell_contours:
        cv2.drawContours(image, [np.array(contour)], -1, (255, 0, 0), 1)
    return image


def handle_image(image_path: str, im: dict) -> go.Figure:
    img = make()
    image_array = read(image_path)
    cell_contours = process_image(image_array, im)
    image_array = add_contours(image_array, cell_contours)
    img = plot(img, image_array)
    if im['selection'] == 'manual':
        img.update_layout(modebar_add=['drawclosedpath', 'eraseshape'])
    return img


def process_image(image_array: npt.NDArray[np.uint8], im: dict) -> list[list[list[int]]]:
    selection = im['selection']
    clickData = im['clickData']
    cell_contours = clickData

    if selection == 'auto':
        clipLimit = im['contrast']
        sq = im['size']
        labels = label(image_array, clipLimit=clipLimit, sq=sq)

        if clickData:
            x, y = clickData[0]
            labels[labels != labels[x, y]] = 0
            labels[labels == labels[x, y]] = 1

        cell_contours = get_contours(labels)

    return cell_contours


def get_ellipse(contour: list[list[int]]) -> cv2.typing.RotatedRect:
    ellipse = cv2.fitEllipse(np.array(contour))
    return ellipse


def cell_info() -> dict:
    info = dict()
    info['eq_diameter'] = 0
    info['ave_diameter'] = 0
    info['perimeter'] = 0
    info['area'] = 0
    info['circularity'] = 0
    info['aspect_ratio'] = 0
    info['taylor_ratio'] = 0
    info['eccentricity'] = 0
    return info


def get_cell_info(contour: list[list[int]], pixel: float = 1.0) -> dict:
    info = cell_info()

    if len(contour) > 1:
        perimeter = cv2.arcLength(np.array(contour), closed=True)
        area = cv2.contourArea(np.array(contour))
        eq_diameter = 2 * np.sqrt(area / np.pi)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        ellipse = get_ellipse(contour)
        _, (d1, d2), _ = ellipse

        d1, d2 = max(d1, d2), min(d1, d2)
        ave_diameter = (d1 + d2) / 2
        aspect_ratio = d1 / d2
        taylor_ratio = (d1 - d2) / (d1 + d2)
        eccentricity = np.sqrt(1 - d2 / d1)

        info['eq_diameter'] = eq_diameter * pixel
        info['ave_diameter'] = ave_diameter * pixel
        info['perimeter'] = perimeter * pixel
        info['area'] = area * (pixel ** 2)
        info['circularity'] = circularity
        info['aspect_ratio'] = aspect_ratio
        info['taylor_ratio'] = taylor_ratio
        info['eccentricity'] = eccentricity

    return info


def correct(angle: float) -> float:
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    return angle


def draw_radius(
    image: npt.NDArray[np.uint8],
    center: tuple[float, float],
    angle: float,
    radius: float,
    color: tuple[int, int, int]
) -> npt.NDArray[np.uint8]:
    xc, yc = center
    x1 = xc + np.cos(np.radians(angle)) * radius
    y1 = yc + np.sin(np.radians(angle)) * radius
    x2 = xc + np.cos(np.radians(angle + 180)) * radius
    y2 = yc + np.sin(np.radians(angle + 180)) * radius
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    return image


def draw_ellipse(
    image: npt.NDArray[np.uint8],
    ellipse: cv2.typing.RotatedRect
) -> npt.NDArray[np.uint8]:
    (xc, yc), (d1, d2), angle = ellipse
    cv2.ellipse(image, ellipse, (0, 255, 0), 1)

    rmajor = max(d1, d2) / 2
    angle = correct(angle)
    image = draw_radius(image, (xc, yc), angle, rmajor, (0, 0, 255))
    rminor = min(d1, d2) / 2
    angle = correct(angle)
    image = draw_radius(image, (xc, yc), angle, rminor, (255, 0, 0))
    return image
