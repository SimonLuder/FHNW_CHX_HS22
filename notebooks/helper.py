import pandas as pd
from datetime import datetime, timedelta

def update_timestamps(df, col):
    '''
    Updates the timestamp of a given column in the dataframe and returns the updated df
    '''
    df["published"] = pd.to_datetime(df["published"], utc=True) + timedelta(hours=2)
    df = df.sort_values(by='published')
    df = df.reset_index(drop=True)
    return df