from pathlib import Path
import pandas as pd

def df_from_file_names(file_names):
    df = pd.DataFrame(file_names, columns=["name"])
    df["label"] = df.name.str.split("/").str[0].astype("category")
    return df

def google_command_organizer(root_path):
    "Creates DataFrame with all files and divisions. Folder structure is assumed to be as here"
    root_path = Path(root_path)
    splits = {}

    already_used_files = set()
    for split, file_name in [("valid", "validation_list.txt"), ("test", "testing_list.txt")]:
        with open(root_path / file_name) as f:
            file_names = f.read().strip().split("\n")
            already_used_files.update(file_names)
            splits[split] = df_from_file_names(file_names)

    training_files = set()
    for dir_ in root_path.iterdir():
        if dir_.is_dir() and dir_.stem != "_background_noise_":
            training_files.update(["/".join(f.parts[-2:]) for f in dir_.iterdir()])

    training_files -= already_used_files

    splits["train"] = df_from_file_names(training_files)

    return splits

            

    
    


    
