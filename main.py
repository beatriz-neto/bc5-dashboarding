# --------------------------------------------------  PACKAGES
import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SUPERHERO],
    suppress_callback_exceptions=True,

)
server = app.server