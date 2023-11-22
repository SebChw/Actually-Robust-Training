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

LOGS_PATH = Path(args.exp_path) / "art_checkpoints"
if not LOGS_PATH.exists():
    raise ValueError(f"Path {LOGS_PATH} does not exist.")
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
STEPS_INFO = prepare_steps_info(LOGS_PATH)
ORDERED_STEPS = [x for x in prepare_steps() if x in STEPS_INFO.keys()]
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
    step_runs_df = STEPS_INFO[step_name][DF]
    step_runs_df.successful = step_runs_df.successful.astype(int)
    conditional = [
        {
            "if": {
                "filter_query": "{successful} = 1",
            },
            "backgroundColor": "rgba(0,255,0,0.2)",
        },
        {
            "if": {
                "filter_query": "{successful} = 0",
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
        step_runs_df.to_dict("records"),
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
    x_attr: str, y_attr: str, step_name: str, selected_row_id: List[int]
) -> px.scatter:
    """Creates figure for the graph."""
    step_runs_df = STEPS_INFO[step_name][DF]
    all_parameters_names = STEPS_INFO[step_name][PARAM_ATTR]
    if selected_row_id is None:
        step_runs_df["size"] = 1
        hover_parameters_names = all_parameters_names
    else:
        selected_row_id = selected_row_id[0]
        run_sizes = [10 if i == selected_row_id else 1 for i in step_runs_df.index]
        step_runs_df["size"] = run_sizes

        step_runs_dict = step_runs_df.to_dict("records")
        selected_run_params = {
            param_name: step_runs_dict[selected_row_id][param_name]
            for param_name in all_parameters_names
        }

        tooltips = []
        for run in step_runs_dict:
            if run["index"] == selected_row_id:
                sample_parameters_names = all_parameters_names
            else:
                sample_parameters_names = [
                    param
                    for param in all_parameters_names
                    if run[param] != selected_run_params[param]
                ]
            tooltips.append(
                "<br>".join(
                    [f"{param}: {run[param]}" for param in sample_parameters_names]
                )
            )

        step_runs_df["description"] = tooltips
        hover_parameters_names = ["description"]

    if x_attr is None and y_attr is None:
        return {}
    elif x_attr is None:
        x_attr = y_attr
    elif y_attr is None:
        y_attr = x_attr

    return px.scatter(
        step_runs_df, x=x_attr, y=y_attr, size="size", hover_data=hover_parameters_names
    )


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
