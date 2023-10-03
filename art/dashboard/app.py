import json
from collections import defaultdict
from pathlib import Path

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
from dash import Dash, Input, Output, callback, dash_table, dcc, html

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

LOGS_PATH = Path("ideal_logs")

steps = defaultdict(lambda: {"full_path": [], "id": [], "results": None, "log": ""})


def merge_steps(step1_dict, step_2_dict):
    # TODO What if we have same steps with different models etc Here I just merge them but best way of doing this should be considered
    step1_dict["full_path"].extend(step_2_dict["full_path"])
    step1_dict["id"].extend(step_2_dict["id"])
    results = step1_dict["results"]

    if results is None:
        results = step_2_dict["results"]
    else:
        results["succesfull"].extend(step_2_dict["results"]["succesfull"])
        results["failed"].extend(step_2_dict["results"]["failed"])
        results["goals"].extend(step_2_dict["results"]["goals"])
    step1_dict["results"] = results

    step1_dict["log"] += step_2_dict["log"]

    return step1_dict


for dir_name in LOGS_PATH.iterdir():
    dir = dir_name.stem.split("_")
    id_, name = dir[1], dir[2]

    with open(dir_name / "results.json", "r") as f:
        results = json.load(f)
    with open(dir_name / "log.txt", "r") as f:
        log = f.read()

    step_dict = {
        "full_path": [dir_name],
        "id": [id_],
        "results": results,
        "log": log,
    }
    steps[name] = merge_steps(steps[name], step_dict)

steps = dict(sorted(steps.items(), key=lambda x: x[1]["id"][0]))


def shorten_result(result):
    return f"n_succesfull: {len(result['succesfull'])}, n_failed: {len(result['failed'])} goal: {result['goals'][0]}"


def create_timeline_item(step_name, result):
    result = result["results"]
    return dmc.TimelineItem(
        title=step_name,
        children=[
            dmc.Text(
                shorten_result(result),
                color="dimmed",
                size="sm",
            ),
        ],
    )


def build_timeline():
    timeline_items = []
    for step_name, result in steps.items():
        timeline_items.append(create_timeline_item(step_name, result))

    return dmc.Timeline(
        active=0,
        bulletSize=15,
        lineWidth=2,
        color="green",
        children=timeline_items,
    )


app.layout = dbc.Container(
    [
        html.H1(children="ART dashboard", style={"textAlign": "center"}),
        dbc.Row(
            [
                # In the future we will build timeline in 2 columns if it is too long
                # dbc.Col(width=3),
                dbc.Col(
                    build_timeline(),
                    width=3,
                ),
                dbc.Col(
                    [
                        html.H3(
                            children="Choose step to visualize",
                            className="text-xl-center text-success p-2",
                        ),
                        dcc.Dropdown(
                            list(steps.keys()),
                            list(steps.keys())[0],
                            id="dropdown-selection",
                            className="p-2 text-center",
                        ),
                        dash_table.DataTable(
                            id="table", data=[], style_table={"overflowX": "scroll"}
                        ),
                    ],
                    width=9,
                ),
            ],
            className="m-5",
        ),
        dbc.Row(
            html.Div(
                id="log-display",
                style={
                    "whiteSpace": "pre-wrap",
                    "padding-top": "15px",
                    "height": "300px",
                    "overflow": "auto",
                },
            )
        ),
        dbc.Row(html.Div([html.H1(children="footer")], id="footer")),
    ]
)


@app.callback(
    [
        Output("table", "data"),
        Output("table", "columns"),
        Output("table", "style_data_conditional"),
    ],
    Input("dropdown-selection", "value"),
)
def updateTable(step_name):
    samples = []
    for status in ["succesfull", "failed"]:
        for sample in steps[step_name]["results"][status]:
            samples.append({**sample, "status": status})
    df = pd.DataFrame(samples)

    conditional = [
        {
            "if": {
                "filter_query": "{status} = succesfull",
            },
            "backgroundColor": "green",
        },
        {
            "if": {
                "filter_query": "{status} = failed",
            },
            "backgroundColor": "tomato",
        },
    ]

    return (
        df.to_dict("records"),
        [{"name": i, "id": i} for i in df.columns],
        conditional,
    )


@callback(
    Output("log-display", "children"),
    Input("dropdown-selection", "value"),
)
def update_log(step_name):
    return steps[step_name]["log"]


if __name__ == "__main__":
    app.run(debug=True)
