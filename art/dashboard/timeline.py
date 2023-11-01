import dash_mantine_components as dmc

from art.dashboard.const import DF


def shorten_result(result):
    n_succsesfull = result["successfull"].sum()
    n_failed = len(result) - n_succsesfull
    models_tried = len(result["model"].unique())
    return f"n_succesfull: {n_succsesfull}, n_failed: {n_failed} models_tried: {models_tried}"


def create_timeline_item(step_name, result):
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


def build_timeline(ordered_steps, steps_info):
    timeline_items = []
    last_successful = 0
    for i, step_name in enumerate(ordered_steps):
        df = steps_info[step_name][DF]
        timeline_items.append(create_timeline_item(step_name, df))
        if df.successfull.any():
            last_successful = i

    return dmc.Timeline(
        active=last_successful,
        bulletSize=15,
        lineWidth=2,
        color="green",
        children=timeline_items,
    )
