from pathlib import Path
import pandas as pd

import torchaudio
from joblib import Parallel, delayed

import art.constants as c


def google_command_organizer(root_path):
    "Creates DataFrame with all files and divisions. Folder structure is assumed to be as here"
    # Alternatively you can add _ before the function and make it outer
    def df_from_file_names(file_names):
        df = pd.DataFrame(file_names, columns=[
                          c.PATH_FIELD], dtype="string")
        df[c.TARGET_FIELD] = df.path.str.split("/").str[0].astype("category")
        df[c.PATH_FIELD] = f"{root_path}/" + df[c.PATH_FIELD]

        return df

    root_path = Path(root_path)
    splits = {}

    already_used_files = set()
    for split, file_name in [(c.VALID_SPLIT, "validation_list.txt"), (c.TEST_SPLIT, "testing_list.txt")]:
        with open(root_path / file_name) as f:
            file_names = f.read().strip().split("\n")
            already_used_files.update(file_names)
            splits[split] = df_from_file_names(file_names)

    training_files = set()
    for dir_ in root_path.iterdir():
        if dir_.is_dir() and dir_.stem != "_background_noise_":
            training_files.update(["/".join(f.parts[-2:])
                                  for f in dir_.iterdir()])

    training_files -= already_used_files

    splits[c.TRAIN_SPLIT] = df_from_file_names(training_files)

    return splits


def cat_to_numeric(splits):
    for df in splits.values():
        cats = df[c.TARGET_FIELD].cat.categories
        df[c.TARGET_FIELD] = df[c.TARGET_FIELD].cat.rename_categories(
            [i for i in range(len(cats))])


def load_entire_dataset(df, load_function=torchaudio.load, n_jobs=-1):

    result = Parallel(n_jobs=n_jobs)(delayed(load_function)(file_path)
                                     for file_path in df.path)

    resultdf = pd.DataFrame(result, columns=["audio", "sr"])

    return pd.concat([df, resultdf], axis=1)
