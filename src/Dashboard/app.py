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
    html.Div(id='save_location', children=[47.5583900, 7.5732700], style={'display': 'none'}),
    html.Div(id='count_map_clicks', children=0, style={'display': 'none'}),
    html.Div(id='save_parking_infos', children={}, style={'display': 'none'}),
    dbc.Row(
        [dbc.Col(
            [html.H2(["infos"]),
            html.Div(children='Kurze Beschreibung wie es funktioniert')],
            width=2,
            style={'margin': "15px"}
        ),
        dbc.Col(
            [html.H1("Intellipark", style={'text-align': 'center'}),
            dl.Map(children=dl.TileLayer(), id="map", center=[47.5588000, 7.5772700], zoom=12.8)],
            style={'height': '80vh', 'margin': "auto"}
        ),
        dbc.Col(
            [html.H2(["Parkhausinfos"]),
            html.Div(id='parkhaus_info')],
            width=2,
            style={'margin': "15px"},
        )]
    )
    ])

@app.callback([Output("map", "children"), Output("save_location", "children"), Output("count_map_clicks", "children"), Output("save_parking_infos", "children")], [Input("count_map_clicks", "children"), Input("map", "click_lat_lng"), Input("save_location", "children")])
def map_click(counter, current_pos, wanted_pos):
    if current_pos is not None:
        counter += 1
        if counter % 2 != 0:
            return [dl.TileLayer([current_pos[1], current_pos[0]]), current_pos, counter, {}]
        else:
            positions_, time_to_parking, parking_name, current_available, predictions = parking_route([current_pos[1], current_pos[0]], [wanted_pos[1], wanted_pos[0]]).get_route_to()
            positions_ = list(map(lambda x: [x[1], x[0]], positions_))
            patterns_ = [dict(offset='0', repeat='1', dash=dict(pixelSize=8, pathOptions=dict(color='#f00', weight=2)))]
            route = dl.PolylineDecorator(positions=positions_, patterns=patterns_)
            return [[dl.TileLayer(), route], wanted_pos, counter, [time_to_parking, parking_name, current_available, predictions]]
    return [dl.TileLayer(), wanted_pos, 0, {}]

@app.callback(Output("parkhaus_info", "children"), [Input("save_parking_infos", "children")])
def update_parkhaus_info(current_pos):
    if current_pos != {}:
        # + heisst es wird mehr freie Parkplätze haben
        trend = '+' if current_pos[2] > current_pos[3] else '-'
        return [html.H4("Parkhaus: " + current_pos[1]),
                html.H4("Freie Parkplätze: " + str(current_pos[2])),
                html.H4("Zeit zum Parkhaus (Min): " + str(round(current_pos[0]/60, 1))),
                html.H4("Trend: " + trend, style={'color': 'green' if trend == '+' else 'red'})]
    return "Keine Parkhausinfos verfügbar"

if __name__ == '__main__':
    app.run_server(debug=True)