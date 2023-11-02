import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Dash, Input, Output, State

from art.dashboard.backend import prepare_steps, prepare_steps_info
from art.dashboard.const import DF, PARAM_ATTR, SCORE_ATTRS
from art.dashboard.layout import get_layout
from art.dashboard.timeline import build_timeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_path",
    type=str,
)
args = parser.parse_args()

LOGS_PATH = Path(args.exp_path) / "checkpoints"
if not LOGS_PATH.exists():
    raise ValueError(f"Path {LOGS_PATH} does not exist.")
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
def updateTable(
    step_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Given selected Step name it returns data for the table.

    Args:
        step_name (str): Name of the step.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]: List of data for the table, columns and style_data_conditional.
    """
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
        Output("radio_x", "options"),
        Output("radio_y", "options"),
    ],
    Input("dropdown-selection", "value"),
)
def update_possible_options(step_name: str) -> Tuple[List[str], List[str]]:
    """Given selected Step name it returns possible options for x and y axis.

    Args:
        step_name (str): Name of the step.

    Returns:
        Tuple[List[str], List[str]] : Possible options for x and y axis.
    """
    scores = STEPS_INFO[step_name][SCORE_ATTRS]
    return scores, scores


@app.callback(
    Output("graph", "figure"),
    [
        Input("radio_x", "value"),
        Input("radio_y", "value"),
        Input("dropdown-selection", "value"),
        Input("table", "selected_rows"),
    ],
)
def update_figure(
    x_attr: str, y_attr: str, step_name: str, selected_row: List[int]
) -> px.scatter:
    """Creates figure for the graph."""
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


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run(debug=True)
