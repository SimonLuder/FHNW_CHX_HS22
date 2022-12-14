import os
import pandas as pd
from datetime import datetime, timedelta


def resample_timestamp(df, t="5Min", how="mean", ignore_cols=None):
    '''
    Resamples the timeseries timestamp per garages to a specified time intervall.
    Args:
        t: intercall, eg: 5Min or 1h
        how: resampling method: max or mean
        ignore_cols: addidional collumns that are ignored in the resampling process and dropped
    Returns:
        Resampled pandas DataFrame
    '''
    num_cols = df._get_numeric_data().columns

    if ignore_cols:
        cat_cols = list(set(df.columns) - set(num_cols) - set(ignore_cols))
    else:
        cat_cols = list(set(df.columns) - set(num_cols))

    df = df.set_index("published")

    if how == "mean":
        df = df.groupby(cat_cols).resample(t).mean()
    elif how == "max":
        df = df.groupby(cat_cols).resample(t).max()
    else:
        print(f"how={how} is not implemented")
        return None

    df = df[num_cols].reset_index()

    return df


def mark_missing_values_windows(df):
    '''
    Creates a new column in the dataframe and marks the missing entry periods
    Args:
        df: pandas DataFrame
    Returns:
        df: updated pandas DataFrame
    '''
    df["missing"] = 0

    df = df.sort_values(by="published").reset_index(drop=True)
    previous_missing = False

    for i, missing_value in enumerate(df.isna().any(axis=1)):
        # if any value is missing
        if missing_value:
            if previous_missing:
                stop = i
            else:
                start = i
                stop = i
                previous_missing = True
        else:
            if previous_missing:
                df.loc[(df.index >= start) & (df.index <= stop), "missing"] = (stop - start + 1)
                previous_missing = False
    return df


def interpolate_historic(df, min_window, ignore_cols=["missing",]):
    '''
    Interpolates the missing values in the dataframe with the historic average
    Args:
        df: pandas DataFrame: dataframe to interpolate on
        min_window: min nr of continuous missing values required for interpolation
        ignore_cols: list of collumns to ignore in interpolation
    Returns:
        df: updated pandas DataFrame
    '''
    df = df.copy()
    df = df.reset_index(drop=True)

    num_cols = df._get_numeric_data().columns
    if ignore_cols is not None:
        num_cols = list(set(num_cols) - set(ignore_cols))

    df['weekday'] = df['published'].dt.dayofweek
    df['time_of_day'] = df['published'].dt.time

    agv_history = df.groupby(["title", "weekday", "time_of_day"]).mean().reset_index()

    for i, row in df.iterrows():
        if row["missing"] >= min_window:
            weekday = row["weekday"]
            time_of_day = row["time_of_day"]
            title = row["title"]
            values = agv_history.loc[(agv_history["weekday"]==weekday) & (agv_history["time_of_day"]==time_of_day) & (agv_history["title"]==title)]
            for c in num_cols:
                df[c][i] = values[c]

    df = df.drop(columns=["weekday", "time_of_day" ])
    return df

def update_timestamps(df):
    '''
    Sets the timestamp to the pandas datetime format
    Args:
        df: pandas DataFrame
    Returns:
        df: updated pandas DataFrame
    '''
    df["published"] = pd.to_datetime(df["published"], utc=True) + timedelta(hours=2)
    df = df.sort_values(by='published')
    df = df.reset_index(drop=True)
    return df


class Interpolator:
    '''
    Interpolator class for small windows of missing values
    '''

    def __init__(self):
        pass


    def interpolate(self, df, method="linear", t="5Min"):
        '''
        This method is used to start the interpolation
        '''

        df = df.copy()
        df = df.drop_duplicates()

        # sort by datetime
        df["published"] = df["published"].apply(pd.to_datetime)
        df = df.sort_values(by='published')

        #call interpolation method
        if method == "linear":
            df = self.__interpolate_scipy(df, "slinear", t)
        else:
            df = self.__interpolate_scipy(df, method, t)
        return df


    def __interpolate_scipy(self, df, method, t, categorical_columns=["title", "name", "id2"]):

        df = df.set_index("published")
        df_intp = None
        for title in set(df["title"]):
            sub_df = df.loc[df["title"]==title].copy()

            # interpolate timestamp
            sub_df = sub_df.resample(t).mean()

            # interpolate numeric values
            sub_df = sub_df.interpolate(method=method)

            # add cathegorical values
            for col in categorical_columns:
                sub_df[col] = df.loc[df["title"]==title][col].iat[0]

            # concat dataframes
            if df_intp is not None:
                df_intp = pd.concat([df_intp, sub_df])
            else:
                df_intp = sub_df
        df_intp = df_intp.reset_index()
        return df_intp

        return df
