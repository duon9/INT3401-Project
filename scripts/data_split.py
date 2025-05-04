import numpy as np
import pandas as pd

def split_by_year(X: np.ndarray, df: pd.DataFrame):
    assert X.shape[0] == len(df), "X và df không cùng số lượng mẫu"

    df = df.reset_index(drop=True)

    year_split = {
        'train': list(range(2010, 2018)),
        'val': [2018],
        'test': [2019,2020]
    }

    result = {}
    for split_name, years in year_split.items():
        indices = df[df['year'].isin(years)].index.to_numpy()
        result[split_name] = (X[indices], df.iloc[indices].reset_index(drop=True))
    
    return result