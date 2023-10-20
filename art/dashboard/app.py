import json
from pathlib import Path
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback
from art.dashboard.timeline import build_timeline
from art.dashboard.backend import prepare_dframes, prepare_steps 
from art.dashboard.layout import get_layout
import plotly.express as px

LOGS_PATH = Path("ideal_logs")
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
ORDERED_STEPS = prepare_steps(LOGS_PATH)
OUTER_DFS, INNER_DFS = prepare_dframes(LOGS_PATH)
TIMELINE = build_timeline(ORDERED_STEPS, OUTER_DFS)

app.layout = get_layout(ORDERED_STEPS, TIMELINE)

WANTED_OUTER_COLUMNS = ["model_hash", "commit_hash", "model", "best_run"]

@app.callback(
    [
        Output("table", "data"),
        Output("table", "columns"),
        Output("table", "style_data_conditional"),
    ],
    Input("dropdown-selection", "value"),
)
def updateTable(step_name):
    df = OUTER_DFS[step_name]
    df.successfull = df.successfull.astype(int)

    conditional = [
        {
            "if": {
                "filter_query": "{successfull} = 1",
            },
            "backgroundColor": "rgba(0,255,0,0.2)",
        },
        {
            "if": {
                "filter_query": "{successfull} = 0",
            },
            "backgroundColor": "rgba(255,0,0,0.2)",
        },
    ]

    columnDefs = [{"name": i, "id": i} for i in WANTED_OUTER_COLUMNS]
    return (
        df.to_dict("records"),
        columnDefs,
        conditional,
    )

@callback(
    Output("checklist1", "options"),
    Input("dropdown-selection", "value"),
)
def update_possible_options(step_name):
    df = INNER_DFS[step_name]
    return list(df.metric.unique())

@callback(
    Output("graph", "figure"),
    [Input("checklist1", "value"),
    Input("dropdown-selection", "value"),]
)
def update_figure(metric_names, step_name):
    df = INNER_DFS[step_name]
    df = df[df.metric.isin(metric_names)]
    return px.scatter(df, x='timestamp', y='value', color='model', symbol='metric')


# @callback(
#     Output("log-display", "children"),
#     Input("dropdown-selection", "value"),
# )
# def update_log(step_name):
#     return steps[step_name]["log"]


if __name__ == "__main__":
    app.run(debug=True)
