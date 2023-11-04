from typing import List

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import dash_table, dcc, get_asset_url, html

from art.dashboard.const import PAD_STYLE, RADIUS_AROUND_STYLE
from art.dashboard.help import HELP


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
                    f"What to put on the {axis} axis",
                    className="text-xl-center fs-5",
                ),
                width=3,
            ),
            dbc.Col(
                dcc.RadioItems(
                    options=[],
                    inline=True,
                    id=f"radio_{axis}",
                    className="fs-5 text-center",
                    style={
                        "display": "flex",
                        "justify-content": "left",
                        "gap": "20px",
                        "align-items": "center",
                    },
                    inputStyle={"margin-right": "10px"},
                )
            ),
        ]
    )


def get_navbar():
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(
                                    src=get_asset_url("pallet.jpg"), height="60ex"
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                html.Div(
                                    "Actually Robust Training",
                                    className="text-xl-center fs-5 fw-bold",
                                    style={"color": "#a87332"},
                                ),
                                width=9,
                                align="center",
                            ),
                        ],
                        justify="center",
                        align="left",
                    )
                ),
                dbc.Button("Help", id="open", n_clicks=0),
                dbc.Modal(
                    [
                        dbc.ModalHeader(
                            dbc.ModalTitle("Welcome to the Art Dashboard"),
                            style={"align": "center"},
                        ),
                        dbc.ModalBody(HELP),
                        dbc.ModalFooter(dbc.Button("Close", id="close", n_clicks=0)),
                    ],
                    id="modal",
                    size="xl",
                    scrollable=True,
                    is_open=False,
                ),
            ]
        ),
        color="#FFFFFF",
        class_name="m-3",
        style=RADIUS_AROUND_STYLE,
    )


def get_layout(ordered_steps: List[str], timeline: dmc.Timeline) -> dbc.Container:
    """Given ordered steps and timeline it returns layout of the dashboard"""
    return dbc.Container(
        [
            get_navbar(),
            dbc.Row(
                [
                    dbc.Col(timeline, width=3, style=RADIUS_AROUND_STYLE | PAD_STYLE),
                    dbc.Col(
                        [
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
                        style=RADIUS_AROUND_STYLE | PAD_STYLE,
                        width=9,
                    ),
                ],
                justify="between",
                class_name="m-3",
            ),
            dbc.Stack(
                [
                    get_radio("x"),
                    get_radio("y"),
                    dbc.Row(dcc.Graph(id="graph")),
                ],
                style=RADIUS_AROUND_STYLE | PAD_STYLE,
                class_name="m-3",
            ),
        ]
    )
