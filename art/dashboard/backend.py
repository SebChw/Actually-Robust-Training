import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

from art.dashboard.const import DF, PARAM_ATTR, SCORE_ATTRS


def prepare_steps_info(logs_path: Path) -> Dict[str, Dict]:
    """Given logs path it returns dictionary with steps information.

    Args:
        logs_path (Path): Path to the logs.

    Returns:
        Dict[str, Dict]: Dictionary with steps information.
    """
    steps_info = defaultdict(
        lambda: {
            SCORE_ATTRS: [],
            PARAM_ATTR: [],
            DF: [],
        }
    )
    for path in Path(logs_path).glob("*/*.json"):
        with open(path) as f:
            step_info = json.load(f)
            step_name = step_info["name"]
            step_model = step_info["model"]
            for run in step_info["runs"]:
                if "regularize" in run["parameters"]:
                    run["parameters"]["regularize"] = stringify_regularize(
                        run["parameters"]["regularize"]
                    )

                new_sample = {
                    "model": step_model,
                    **run["scores"],
                    **run["parameters"],
                    "timestamp": run["timestamp"],
                    "hash": run["hash"],
                    "commit_id": run["commit_id"],
                    "successful": run["successful"],
                }
                steps_info[step_name][DF].append(new_sample)
                steps_info[step_name][SCORE_ATTRS] = list(run["scores"].keys())
                steps_info[step_name][PARAM_ATTR] = list(run["parameters"].keys()) + [
                    "commit_id",
                    "hash",
                ]

    for step_info in steps_info.values():
        step_info[DF] = pd.DataFrame(step_info[DF])
        step_info[DF] = step_info[DF].reset_index()

    return steps_info


def stringify_regularize(regularize: Dict) -> str:
    """Since regularize field contain list we must handle them with special care        .

    Args:
        regularize (Dict): regularize field from results.json

    Returns:
        str: stringified version of regularize field
    """
    parameters = []
    for key, value in regularize.items():
        if key in ["model_modifiers", "datamodule_modifiers"]:
            continue
        parameters.append(f"{key}={value}")
    representation = ""
    if parameters:
        representation += f"model-kwargs={' '.join(parameters)} |"
    if regularize["model_modifiers"]:
        representation += f"model-modifiers={regularize['model_modifiers']} |"
    if regularize["datamodule_modifiers"]:
        representation += f"datamodule-modifiers={regularize['datamodule_modifiers']}"
    return representation


def prepare_steps():
    return [
        "Data analysis",
        "Evaluate Baseline",
        "Check Loss On Init",
        "Overfit One Batch",
        "Overfit",
        "TransferLearning",
        "Regularize",
    ]
