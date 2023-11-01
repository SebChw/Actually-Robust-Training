import argparse
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Dash, Input, Output

from art.dashboard.backend import prepare_steps, prepare_steps_info
from art.dashboard.const import DF, PARAM_ATTR, SCORE_ATTRS
from art.dashboard.layout import get_layout
from art.dashboard.timeline import build_timeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--logs_path",
    type=str,
    default="../../../art_template/{{cookiecutter.project_slug}}/exp1/checkpoints",
)
args = parser.parse_args()

LOGS_PATH = Path(args.logs_path)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
ORDERED_STEPS = prepare_steps(LOGS_PATH)
STEPS_INFO = prepare_steps_info(LOGS_PATH)
TIMELINE = build_timeline(ORDERED_STEPS, STEPS_INFO)
app.layout = get_layout(ORDERED_STEPS, TIMELINE)


@app.callback(
    [
        Output("table", "data"),
        Output("table", "columns"),
        Output("table", "style_data_conditional"),
    ],
    Input("dropdown-selection", "value"),
)
def updateTable(step_name):
    df = STEPS_INFO[step_name][DF]
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

    ordered_names = (
        STEPS_INFO[step_name][SCORE_ATTRS]
        + STEPS_INFO[step_name][PARAM_ATTR]
        + ["timestamp"]
    )
    columnDefs = [{"name": i, "id": i} for i in ordered_names]
    return (
        df.to_dict("records"),
        columnDefs,
        conditional,
    )


@app.callback(
    [
        Output("checklist_x", "options"),
        Output("checklist_y", "options"),
    ],
    Input("dropdown-selection", "value"),
)
def update_possible_options(step_name):
    scores = STEPS_INFO[step_name][SCORE_ATTRS]
    return scores, scores


@app.callback(
    Output("graph", "figure"),
    [
        Input("checklist_x", "value"),
        Input("checklist_y", "value"),
        Input("dropdown-selection", "value"),
        Input("table", "selected_rows"),
    ],
)
def update_figure(x_attr, y_attr, step_name, selected_row):
    df = STEPS_INFO[step_name][DF]
    parameters = STEPS_INFO[step_name][PARAM_ATTR]
    if selected_row is None:
        df["size"] = 1
        hover_params = parameters
    else:
        selected_row = selected_row[0]
        df["size"] = [10 if i == selected_row else 1 for i in df.index]

        tooltips = []
        samples = df.to_dict("records")
        selected_params = {key: samples[selected_row][key] for key in parameters}
        for row in samples:
            if row["index"] == selected_row:
                params = parameters
            else:
                params = [
                    param
                    for param in parameters
                    if row[param] != selected_params[param]
                ]
            tooltips.append("<br>".join([f"{param}: {row[param]}" for param in params]))

        df["description"] = tooltips
        hover_params = ["description"]

    if x_attr is None and y_attr is None:
        return {}
    elif x_attr is None:
        x_attr = y_attr
    elif y_attr is None:
        y_attr = x_attr

    return px.scatter(df, x=x_attr, y=y_attr, size="size", hover_data=hover_params)


if __name__ == "__main__":
    app.run(debug=True)
