from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

from src.components.layout import create_layout


def run() -> None:
    app = Dash(external_stylesheets=[BOOTSTRAP])
    server = app.server
    app.title = "AFM Curve Analysis"
    app.layout = create_layout(app)
    app.run(debug=True)


if __name__ == "__main__":
    run()
