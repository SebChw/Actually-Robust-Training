import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


def get_layout(ordered_steps, timeline):
    return dbc.Container(
    [
        html.H1(children="ART dashboard", style={"textAlign": "center"}),
        dbc.Row(
            [
                # In the future we will build timeline in 2 columns if it is too long
                # dbc.Col(width=3),
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
                            id="table", data=[], style_table={"overflowX": "scroll"},row_selectable="multi",
                        ),
                    ],
                    width=9,
                ),
            ],
            className="m-5",
        ),
        dbc.Row(
            [
                dbc.Col(html.P("Choose stuff to compare:", className="text-xl-center text-success p-2")),
            dbc.Col(dcc.Checklist(['New York City', 'Montreal','San Francisco'], [], inline=True, id='checklist1'))
            ]),
        dbc.Row(
            [
                dbc.Col(html.P("Choose x axis:", className="text-xl-center text-success p-2")),
            dbc.Col(dcc.Checklist(['New York City', 'Montreal','San Francisco'], [], inline=True, id='checklist2'))
            ],
        ),
        dbc.Row(
            [
                dbc.Col(html.P("Choose even more stuff:", className="text-xl-center text-success p-2")),
            dbc.Col(dcc.Checklist(['New York City', 'Montreal','San Francisco'], [], inline=True, id='checklist3'))
            ]
            
        ),
        dbc.Row(
            dcc.Graph(id='graph')
        ),

        #TODO think on how to show different assets than plots in one place
        # dbc.Row(
        #     html.Div(
        #         id="log-display",
        #         style={
        #             "whiteSpace": "pre-wrap",
        #             "padding-top": "15px",
        #             "height": "300px",
        #             "overflow": "auto",
        #         },
        #     )
        # ),

        dbc.Row(html.Div([html.H1(children="footer")], id="footer")),
    ]
)