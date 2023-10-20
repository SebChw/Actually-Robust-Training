from pathlib import Path
from art.step.step_state import parse_step_trials, dataframemize
import json

def prepare_dframes(logs_path):
    step_trials = []
    for path in Path(logs_path).glob("*/*.json"):
        with open(path) as f:
            step_trials.append(parse_step_trials(json.load(f)))

    outer_dfs, inner_dfs = dataframemize(step_trials)
    return outer_dfs, inner_dfs

def prepare_steps(logs_path):
    # TODO to order which steps were first is another issue for me. Should we stick to the integer inside a directory name?
    return ["Evaluate Baseline", "Check Loss On Init"]

