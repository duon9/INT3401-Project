import pandas as pd
import os

def parse_filename(filename : str):

    base_name = os.path.basename(filename)
    
    if "POSITIVE" in base_name:
        id_part = base_name.split("_")[1].replace(".nc", "")
        label = 0
    elif "NEGATIVE" in base_name:
        parts = base_name.split("_")
        id_part = parts[1]
        label = int(parts[2])

    return id_part, label

def create_metadata(folder_path : str):
    data = []
    lst = sorted(os.listdir(folder_path))
    for file_name in lst:
        if file_name.endswith(".nc"):
            file_path = os.path.join(folder_path, file_name)
            data.append(parse_filename(file_path))
    
    df = pd.DataFrame(data, columns=["id", "label"])
    return df