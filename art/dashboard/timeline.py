from typing import Dict, List

import dash_mantine_components as dmc
import pandas as pd

from art.dashboard.const import DF


def shorten_result(result: pd.DataFrame) -> str:
    """Shorten result of a Step to one line.

    Args:
        result (pd.DataFrame): All runs of a Step.

    Returns:
        str: Shortened result.
    """
    n_succsesfull = result["successful"].sum()
    n_failed = len(result) - n_succsesfull
    models_tried = len(result["model"].unique())
    return (
        f"Succesfull: {n_succsesfull}, Failed: {n_failed} models Used: {models_tried}"
    )


def create_timeline_item(step_name: str, result: pd.DataFrame) -> dmc.TimelineItem:
    """Given step name and dataframe with all step runs it returns timeline item."""
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


def build_timeline(
    ordered_steps: List[str], steps_info: Dict[str, Dict]
) -> dmc.Timeline:
    """Given ordered steps and steps information it returns timeline."""
    timeline_items = []
    last_successful = 0
    for i, step_name in enumerate(ordered_steps):
        df = steps_info[step_name][DF]
        timeline_items.append(create_timeline_item(step_name, df))
        if df.successful.any():
            last_successful = i

    return dmc.Timeline(
        active=last_successful,
        bulletSize=15,
        lineWidth=2,
        color="green",
        children=timeline_items,
    )
