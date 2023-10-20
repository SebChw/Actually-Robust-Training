from typing import List, Dict
from dataclasses import dataclass
import pandas as pd

@dataclass
class Score:
    metric: str
    state: str
    value: float


@dataclass
class StepState:
    model_hash: str
    scores: List[Score]
    commit_hash: str
    successfull: bool
    timestamp: int

    def outer_info(self, other: Dict):
        return {
            'model_hash': self.model_hash,
            'commit_hash': self.commit_hash,
            'successfull': self.successfull,
            "timestamp": self.timestamp,
            **other,
        }
    
    def inner_info(self, other: Dict):
        return  [{**score.__dict__, **self.outer_info(other)} for score in self.scores]  

@dataclass
class StepTrials:
    step: str
    model: str
    best_run: StepState
    other_runs: List[StepState]
    goals: List

    def outer_generator(self,):
        yield self.best_run.outer_info({'model': self.model, "best_run": True})
        for run in self.other_runs:
            yield run.outer_info({'model': self.model, "best_run": False}) 
        
    def outer_list(self,):
        return list(self.outer_generator())
    
    def inner_generator(self,):
        yield self.best_run.inner_info({'model': self.model, "best_run": True})
        for run in self.other_runs:
            yield run.inner_info({'model': self.model, "best_run": False})
    
    def inner_list(self,):
        inner_list = []
        for element in self.inner_generator():
            inner_list.extend(element)
        return inner_list

def parse_step_state(step_state: Dict):
    step_state['scores'] = [Score(**score) for score in step_state['scores']]
    return StepState(**step_state)
    

def parse_step_trials(json: Dict) -> StepTrials:
    json['best_run'] = parse_step_state(json['best_run']) 
    json['other_runs'] = [parse_step_state(run) for run in json['other_runs']]
    return StepTrials(**json)

from collections import defaultdict

def dataframemize(steps_trials: List[StepTrials]):
    outer_df = defaultdict(lambda : [])
    inner_dfs = defaultdict(lambda : []) 
    for step_trials in steps_trials:
        outer_df[step_trials.step].extend(step_trials.outer_list())
        inner_dfs[step_trials.step].extend(step_trials.inner_list())

    outer_df = {k: pd.DataFrame(v) for k, v in outer_df.items()}
    #print(inner_dfs)
    inner_dfs = {k: pd.DataFrame(v) for k, v in inner_dfs.items()}
    return outer_df, inner_dfs




