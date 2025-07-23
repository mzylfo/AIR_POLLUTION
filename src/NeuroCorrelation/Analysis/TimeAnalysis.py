import datetime
from datetime import timedelta
from pathlib import Path
import pandas as pd

class TimeAnalysis():
    
    def __init__(self, folder_path):
        self.time_dict = dict()
        self.folder_path = folder_path
        self.time_df = pd.DataFrame(columns=["name", "time", "values"])
        
    def start_time(self, name):
        if name not in self.time_dict:
            self.time_dict[name] = {'start_time':[], 'end_time':[], 'time_diff':[]}
        self.time_dict[name]['start_time'].append(datetime.datetime.now())
        
    def stop_time(self, name):
        self.time_dict[name]['end_time'].append(datetime.datetime.now())
        self.time_dict[name]['time_diff'].append(self.time_dict[name]['end_time'][-1] - self.time_dict[name]['start_time'][-1])
    
    def compute_time(self, name, fun= "diff"):
        temp_df = pd.DataFrame(self.time_dict[name])
        #temp_df['start_time'] = pd.to_datetime(temp_df['start_time'])
        #temp_df['end_time'] = pd.to_datetime(temp_df['end_time'])
        #temp_df['time_diff'] = temp_df['end_time'] - temp_df['start_time']

        if fun == "diff" or fun=="first":
            time_val = temp_df['time_diff'][0]
        if fun == "sum":
            time_val = sum(self.time_dict[name]['time_diff'], timedelta())
        if fun == "mean":
            time_val = sum(self.time_dict[name]['time_diff'], timedelta())/len(self.time_dict[name]['time_diff'])
        if fun == "last":
            time_val = temp_df['time_diff'][-1]
            
        row = {"name":name, "time":time_val, "values":temp_df['time_diff'].to_list()}
        self.time_df.loc[len(self.time_df)] = row

    def get_time(self, name, fun="last"):
        
        if fun == "first":
            time_val = self.time_dict[name]['time_diff'][0]
        if fun == "sum":
            time_val = sum(self.time_dict[name]['time_diff'], timedelta())

        if fun == "mean":
            time_val = sum(self.time_dict[name]['time_diff'], timedelta())/len(self.time_dict[name]['time_diff'])
        if fun == "last":
            time_val = self.time_dict[name]['time_diff'][-1]
        
        return time_val
    
    def save_time(self):
        path_file = Path(self.folder_path,"time_values.csv")
        self.time_df.to_csv(path_file, index=False)
        