
import json
from pathlib import Path


class DatasetTool:
    
    def __init__(self, name_dataset, version_dataset, time_slot=None):
        self.name_dataset = name_dataset
        self.version_dataset = version_dataset
        self.time_slot = time_slot
        self.folderpath = Path('src','NeuroCorrelation','Datasets')
        self.filepath = self.set_filepath(name_dataset=self.name_dataset, folderpath=self.folderpath)
        
        
        
    def set_filepath(self, name_dataset, folderpath):
        if name_dataset == "METR_LA":
            filepath = Path(folderpath,"METR_LA_dataset.json")
        elif name_dataset == "PEMS_BAY":
            filepath = Path(folderpath,"PEMS_BAY_dataset.json")
        elif name_dataset == "CHENGDU":
            filepath = Path(folderpath,"CHENGDU_dataset.json")
        elif name_dataset == "SP500":
            filepath = Path(folderpath,"SP500_dataset.json")
        else:
            filepath = Path()
        return filepath
    
    def get_dataset_settings(self):
        return self.load_fileJson(filepath=self.filepath, name_dataset=self.name_dataset, version_dataset=self.version_dataset)

        
    def load_fileJson(self, filepath, name_dataset, version_dataset):
        print("filepath",filepath)
        with open(filepath, 'r') as f:
            config = json.load(f)
            
        
        dataset_settings = dict()
        if config[name_dataset][version_dataset]["filename"] is not None:
            if self.time_slot is not None:
                if config[name_dataset][version_dataset]["filename"][self.time_slot] is not None:
                    dataset_settings["filename"] = Path(config[name_dataset][version_dataset]["filename"][self.time_slot])
            else:
                print(config[name_dataset][version_dataset]["filename"])
                dataset_settings["filename"] = Path(config[name_dataset][version_dataset]["filename"])
        else:
            dataset_settings["filename"] = None
            
        if config[name_dataset][version_dataset]["pathMap"] is not None:
            dataset_settings["pathMap"] = Path(config[name_dataset][version_dataset]["pathMap"])
        else:
            dataset_settings["pathMap"] = None
        
        if config[name_dataset][version_dataset]["edge_path"] is not None:
            dataset_settings["edge_path"] = Path(config[name_dataset][version_dataset]["edge_path"])
        else:
            dataset_settings["edge_path"] = None
            
        if config[name_dataset][version_dataset]["timeweather_path"] is not None:
            dataset_settings["timeweather_path"] = Path(config[name_dataset][version_dataset]["timeweather_path"][self.time_slot])
        else:
            dataset_settings["timeweather_path"] = None
            
        if config[name_dataset][version_dataset]["copula_filename"] is not None:
            if self.time_slot in config[name_dataset][version_dataset]["copula_filename"]:
                if self.time_slot in config[name_dataset][version_dataset]["copula_filename"][self.time_slot] is not None:
                    dataset_settings["copula_filename"] = Path(config[name_dataset][version_dataset]["copula_filename"][self.time_slot])
                else:
                    dataset_settings["copula_filename"] = None
            else:   
                dataset_settings["copula_filename"] = None
        else:
            dataset_settings["copula_filename"] = None
        
        return dataset_settings
        