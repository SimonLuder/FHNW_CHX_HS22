import pandas as pd
import numpy as np

def clean_data(path_to_file: str, file: str) -> dict:
    """
    Function to clean data
    """
    # read data
    df = pd.read_csv(path_to_file + file, sep=';')

    # drop columns
    df = df[['published', 'free', 'id2', 'total', 'geo_point_2d']]

    # set datatypes
    df.published = pd.to_datetime(df.published)
    df['id2'] = df['id2'].astype(str)
    df['geo_point_2d'] = df['geo_point_2d'].astype(str)

    # split geo_point_2d
    df['latitude'] = df.geo_point_2d.apply(lambda x: float(x.strip().split(',')[0]))
    df['longitude'] = df.geo_point_2d.apply(lambda x: float(x.strip().split(',')[1]))
    df.drop('geo_point_2d', axis=1, inplace=True)

    # replace free spots with nan if number is bigger than total spots
    def replace_with_nan(df):
        i = 0
        while i < len(df):
            if df.loc[i, 'free'] > df.loc[i, 'total']:
                df.loc[i, 'free'] = np.nan
                i += 1
            else:
                i += 1
        return df
    df = replace_with_nan(df)
    # delete garage claramatte
    df = df[df['id2'] != 'claramatte']

    # interpolate nans in free spots
    for i in df['id2'].unique():
        df.loc[df['id2'] == i,'free'] = df.loc[df['id2'] == i,'free'].interpolate(method='linear')
    df = df.iloc[1:,:]
    df['free'] = df['free'].round().astype(int)
    return df