from turtle import position
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash import dcc
from dash.dependencies import Input, Output
import dash_leaflet as dl
import dash_bootstrap_components as dbc
from src.Dashboard.function_class import parking_route

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row(
        dbc.Col(html.H1("Hello World")),
        ),
    dbc.Row(
        dbc.Col(dcc.Input(id='input-longitude', value=7.64511, type='text')),
        ),
    dbc.Row(
        dbc.Col(dcc.Input(id='input-latitude', value=47.52271, type='text')),
    ),
    dbc.Row(
        dbc.Col(
            dl.Map(children=dl.TileLayer(), id="map"),
            style={'width': '50%', 'height': '50vh', 'margin': "auto", "display": "block"}
        )
    )
    ])

@app.callback(
    Output("map", "children"),
    [Input("input-latitude", "value"), Input("input-longitude", "value")]
)
def update_map(lat, lon):
    positions_ = parking_route([lat, lon]).get_route_to()
    positions_ = list(map(lambda x: [x[1], x[0]], positions_))
    print(positions_)
    patterns_ = [dict(offset='12', repeat='25', dash=dict(pixelSize=10, pathOptions=dict(color='#f00', weight=2))),
            dict(offset='0', repeat='25', dash=dict(pixelSize=0))]
    route = dl.PolylineDecorator(positions=positions_, patterns=patterns_)
    return [dl.TileLayer(), route]

if __name__ == '__main__':
    app.run_server(debug=True)