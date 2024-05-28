from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

from .src.components.layout import create_layout


def run(debug: bool = False) -> None:
    app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
    app.title = "AFM Data Analysis"
    app.layout = create_layout(app)
    app.run(debug=debug)
