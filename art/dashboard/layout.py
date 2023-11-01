from typing import List

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import dash_table, dcc, html


def get_radio(axis: str = "x") -> dbc.Row:
    """Given axis it returns checklist.

    Args:
        axis (str, optional):. Defaults to "x".

    Returns:
        dbc.Row: Radio items.
    """
    return dbc.Row(
        [
            dbc.Col(
                html.P(
                    f"What to put on {axis} axis",
                    className="text-xl-center text-success p-2",
                )
            ),
            dbc.Col(
                dcc.RadioItems(
                    options=[],
                    inline=True,
                    id=f"radio_{axis}",
                )
            ),
        ]
    )


def get_layout(ordered_steps: List[str], timeline: dmc.Timeline) -> dbc.Container:
    """Given ordered steps and timeline it returns layout of the dashboard"""
    return dbc.Container(
        [
            html.H1(children="ART dashboard", style={"textAlign": "center"}),
            dbc.Row(
                [
                    dbc.Col(
                        timeline,
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.H3(
                                children="Choose step to visualize",
                                className="text-xl-center text-success p-2",
                            ),
                            dcc.Dropdown(
                                ordered_steps,
                                ordered_steps[0],
                                id="dropdown-selection",
                                className="p-2 text-center",
                            ),
                            dash_table.DataTable(
                                id="table",
                                data=[],
                                style_table={"overflowX": "scroll"},
                                row_selectable="single",
                                filter_action="native",
                                sort_action="native",
                                sort_mode="multi",
                            ),
                        ],
                        width=9,
                    ),
                ],
                className="m-5",
            ),
            get_radio("x"),
            get_radio("y"),
            dbc.Row(dcc.Graph(id="graph")),
            dbc.Row(html.Div([html.H1(children="footer")], id="footer")),
        ]
    )
