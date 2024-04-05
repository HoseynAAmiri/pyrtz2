from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

from .src.components.layout import create_layout


def run() -> None:
    app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
    app.title = "AFM Curve Analysis"
    app.layout = create_layout(app)
    app.run(debug=False)


if __name__ == "__main__":
    run()
