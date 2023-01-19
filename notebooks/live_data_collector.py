import requests
import pandas as pd
import json
import pprint
import os
import time
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")




class LiveDataCollector():
    
    def __init__(self):
        self.update_time = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=1)

        
    def __open_csv(self, file_path):
        '''
        Checks if a csv exists and returns it as dataframe. 
        If no source was found it return a empty dataframe.
        Args: 
            file_path (str): path of file
        Returns: 
            df (pandas DataFrame): Loaded csv as dataframe
        '''
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame()
        return df


    def __save_csv(self, df, file_path):
        '''
        Saves a dataframe as csv at a given file path
        Args: 
            df (pandas DataFrame): dataframe which needs to be saved
            file_path (str): path of file            
        '''
        df.to_csv(file_path, index=False)
        
    def __call_newest_data(self, url):
        '''
        Requests the newest data from the data.bs.ch api and returns it as pandas dataframe.
        Args:
            url (str): url where the api is located
        '''
        try:
            r = requests.get(url)
            df = pd.DataFrame(r.json())
            df = df.dropna(axis="rows")
            return df
        except requests.exceptions.RequestException as e: 
            return None
    
    
    def collect_data(self, url, file_path, verbose=False):
        '''
        Main method that is called for the collection and saving of new data
        Args: 
            url (str): url where the api is located
            file_path (str): local path to save the data
            verbose: (bool): if true prints when new data gets saved
        '''
        # get current datetime
        now = datetime.now()
        now = now.replace(second=0, microsecond=0)

        # check if last update is older than a minute
        if (self.update_time < now): 
            
            # import csv as dataframe
            df = self.__open_csv(file_path)

            # get new data
            df_new = self.__call_newest_data(url)
            
            if df_new is not None:
                # update dataframe
                df = df.append(df_new)
                df = df.reset_index(drop=True)

                # save as csv
                if verbose:
                    print(f"Saving: {file_path} at time: {now}")
                self.__save_csv(df, file_path)
                if verbose:
                    print("Complete")

                # update update time
                self.update_time = now
                
if __name__ == '__main__':      
    
    url = "https://data.bs.ch/api/v2/catalog/datasets/100088/exports/json?limit=-1&offset=0&timezone=Europe%2FBerlin"
    path = "../data/"

    collector = LiveDataCollector()

    while True:
        # define csv name
        file = f'{(datetime.now().strftime("%d_%m_%Y"))}.csv'
        # update csv
        collector.collect_data(url, path+file, verbose=True)
        time.sleep(10)
        