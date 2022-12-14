import requests
import pandas as pd
import json
import pprint
import os
import time
from datetime import datetime, timedelta
from collect_live_data import LiveDataCollector
from helper import update_timestamps, resample_timestamp, interpolate_historic, Interpolator

import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':

    # setup ------------------------------------------------------------------------- #
    resampling = "5min"
    path = "../data/dashboard/"
    unprocessed_csv = "unprocessed.csv"
    processed_csv = "processed_{}.csv".format(resampling)
    url = "https://data.bs.ch/api/v2/catalog/datasets/100088/exports/json?limit=-1&offset=0&timezone=Europe%2FBerlin"  #api
    # ------------------------------------------------------------------------------- #

    # create dashboard data dir of doesnt exist
    if not os.path.exists(path):
        os.makedirs(path)

    # run live data collector
    collector = LiveDataCollector()

    # collect data
    last_processing = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=1)

    while True:

        # call api
        collector.collect_data(url, path+unprocessed_csv, verbose=True)

        # check if new data has been downloaded
        if collector.update_time > last_processing:

            # load unprocessed data
            df = pd.read_csv(path+unprocessed_csv)
            df = df.drop_duplicates()
            df = update_timestamps(df)

            # drop entries older then 1 day from unprocessed_csv
            df = df[df["published"].dt.tz_localize(None) > datetime.now() - timedelta(days=1)]
            df.to_csv(path + processed_csv, index=False)

            # drop irrelevant columns
            df = df.drop(columns=["link", "geo_point_2d", "description"])

            # resample time
            df = resample_timestamp(df, t=resampling, how="mean", ignore_cols=['published', "description"])

            # interpolate missing values if there are any
            if len(df.loc[df.isna().any(axis=1)]) > 0:
                df = df.groupby(by=["id2", "name", "title"]).apply(mark_missing_values_windows)
                df = df.reset_index(drop=True)
                df = interpolate_historic(df, 9)

                ip = Interpolator()
                df = ip.interpolate(df, method="linear", t=resampling)

            # save procesed
            df.to_csv(path + processed_csv, index=False)

            # update timestamp
            last_processing = collector.update_time
