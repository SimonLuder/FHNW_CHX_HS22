from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row(
        dbc.Col(html.H1("Hello World"))
        ),
    dbc.Row(
        dbc.Col(
            html.Iframe(src="https://www.google.com/maps/embed/v1/place?key=AIzaSyAdcC818Yu5o4z1vlE-KJ-za9TT-TuUCdc&q=Space+Needle,Basel", width="600", height="450", style={'border':0})
                        ),
        dbc.Col(
            html.Div(dcc.Input(id='input-on-submit', type='text'))
        )
        ),
    dbc.Row(
        dbc.Col(
            html.Div(id='output-container')
        )
    )
        
        ])

@app.callback(
    Output('output-containerc', 'children'),
    Input('input-on-submit', 'text'))
def update_output(value):
    return 'test' + value


if __name__ == '__main__':
    app.run_server(debug=True)