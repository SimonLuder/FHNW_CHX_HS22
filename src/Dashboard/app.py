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

# @app.callback(
#     Output("map", "children"),
#     [Input("input-latitude", "value"), Input("input-longitude", "value")]
# )
# def update_map(lat, lon):
    # positions_ = parking_route([lon, lat]).get_route_to()
@app.callback(Output("map", "children"), [Input("map", "click_lat_lng")])
def map_click(click_lat_lng):
    if click_lat_lng is not None:
        positions_ = parking_route([click_lat_lng[1], click_lat_lng[0]]).get_route_to()
        positions_ = list(map(lambda x: [x[1], x[0]], positions_))
        patterns_ = [dict(offset='0', repeat='1', dash=dict(pixelSize=8, pathOptions=dict(color='#f00', weight=2)))]
        route = dl.PolylineDecorator(positions=positions_, patterns=patterns_)
        return [dl.TileLayer(), route]
    else:
        return [dl.TileLayer()]

if __name__ == '__main__':
    app.run_server(debug=True)