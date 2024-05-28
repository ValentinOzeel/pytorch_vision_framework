import os 
import yaml
from typing import List 

import torch
from torchvision import datasets, transforms
import torchmetrics

import gzip
import shutil
from colorama import init, Fore, Back, Style
init() # Initialize Colorama to work on Windows

from .datasets import CustomDfImageFolder

# Assuming data_exploration.py is in src\main.py
#project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#config_path = os.path.join(project_root_path, 'conf', 'config.yml')


class ConfigLoad():
    def __init__(self, path:str):
        self.path = path
        with open(self.path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def get_config(self):
        return self.config
            
    def get_transform_steps(self, dict_name='DATA_TRANSFORM_AND_AUGMENTATION', dataset_type='train') -> List:
        '''
        Access transformation dict defined in config
        Transform it as a list of torchvision.transforms steps
        '''
        yml_dict = self.config[dict_name][dataset_type.lower()]
        steps = []
        for step_name, params in yml_dict.items():
            # Get the transforms method
            transform_step = getattr(transforms, step_name)
            # Initialize the transform method with its defined parameters and append in list
            if params: 
                steps.append(transform_step(**params))
            # Just add a Normalize flag: Normalization will be computed on training dataset upon Dataset creation
            elif step_name.lower() == 'normalize':
                steps.append('Normalize')
            else:
                steps.append(transform_step()) 
        return steps
    
    def get_dataset(self, dict_name='DATASET'):
        return CustomDfImageFolder if self.config[dict_name] == 'CustomDfImageFolder' else getattr(datasets, self.config[dict_name])
    
    def get_nested_param(self, config_dict:dict):
        return next(iter(config_dict.items()))
        
    def get_torchmetrics_dict(self, device:str, dict_name='torchmetrics'):
        config = self.config['MODEL_PARAMS'][dict_name]
        return {metric_name:getattr(torchmetrics, metric_name)(**params).to(device) for metric_name, params in config.items()}


def check_cuda_availability():
    is_or_is_not = 'is' if torch.cuda.is_available() else 'is not'
    symbol = '✔' if torch.cuda.is_available() else 'X'
     
    print(f"{symbol*2} --- Cuda {is_or_is_not} available on your machine. --- {symbol*2}")
    
    
def colorize(to_print, color):
    return f"{getattr(Fore, color) + to_print + Style.RESET_ALL}"


def decompress_model(model_path:str):
    output_path = model_path.replace('.gz', '') if '.gz' in model_path else model_path
                                              
    # Decompress the model file
    with gzip.open(model_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)