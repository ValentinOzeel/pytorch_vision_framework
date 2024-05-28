import os
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image 
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold



class DataPrep():
    '''
    A class for preparing datasets by creating DataFrames (columns = path and class), splitting data into training, validation, and test sets,
    and generating cross-validation splits.

    Parameters:
    - root (str): Path to the root directory containing the image data.
    - random_seed (int, optional): Seed for random number generation to ensure reproducibility. Default is None.
    
    Attributes:
    - all_img_paths_in_root (List[Path]): List of all image paths (jpg, png) in the root directory.
    - random_seed (int): Seed for random number generation to ensure reproducibility.
    - original_df (pd.DataFrame): DataFrame containing the paths and class labels of the original dataset.
    - train_df (pd.DataFrame): DataFrame containing the training data after splitting.
    - test_df (pd.DataFrame): DataFrame containing the test data after splitting.
    - val_df (pd.DataFrame): DataFrame containing the validation data after splitting.
    - cv_indices (Dict[int, Dict[str, pd.DataFrame]]): Dictionary containing cross-validation data splits.
    '''
    def __init__(self, root:str, random_seed:int=None):
        
        if not os.path.exists(root): raise FileNotFoundError(f"The root path is not valid: {root}")
        # List all Path of .jpg and .png images in root dir (expect following path format: root/class/image.jpg)
        self.all_img_paths_in_root = list(Path(root).glob("**/*.jpg")) + list(Path(root).glob("**/*.png"))
        self.random_seed = random_seed
        
    def create_path_class_df(self):
        '''
        Create a DataFrame containing image paths and their corresponding class labels.
        Returns: - pd.DataFrame: DataFrame with columns ['path', 'class'].
        '''
        self.original_df = pd.DataFrame(
            [(img_path, img_path.parent.name) for img_path in self.all_img_paths_in_root],
            columns=['path', 'class']
            )
        return self.original_df
         
    def train_test_presplit(self, train_ratio:float):
        '''
        Pre-split the original DataFrame into training and test sets based on the specified train_ratio.
        Parameters:  - train_ratio (float): Proportion of the data to be used for training. Must be between 0 and 1.
        Returns: - Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for the training and test sets.
        '''
        if hasattr(self, 'val_df') or hasattr(self, 'cv_indices'):
            raise ValueError("Cannot run 'train_test_presplit' method after 'train_valid_split' nor 'cv_splits' methods.")
        if not hasattr(self, 'original_df'):
            raise ValueError("User must create the original df with the 'create_df' method before being able to run the 'train_test_presplit' method.")
        
        if train_ratio > 1 or train_ratio < 0:
            raise ValueError("train_test_presplit's train_ratio parameter must be a float comprised between 0 and 1")
        
        X, y = self.original_df.drop(columns=['class']), self.original_df['class']
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=self.random_seed)
        # Get positions of train and test data in original df
        train_positions, test_positions = next(sss.split(X, y))
        # Effectively perform the presplit while preserving initial indices
        self.train_df = self.original_df.iloc[train_positions]
        self.test_df = self.original_df.iloc[test_positions]
        
        return self.train_df, self.test_df


    def train_valid_split(self, train_ratio:float):
        '''
        Split the training data into training and validation sets based on the specified train_ratio.
        Parameters: - train_ratio (float): Proportion of the data to be used for training. Must be between 0 and 1.
        Returns: - Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for the training and validation sets.
        '''
        if train_ratio > 1 or train_ratio < 0:
            raise ValueError("train_valid_presplit's train_ratio parameter must be a float comprised between 0 and 1")
        
        # If class has self.train_df, it means a split has already been done so we consider this, otherwise we consider the original df 
        train_df = self.train_df if hasattr(self, 'train_df') else self.original_df
        
        X, y = train_df.drop(columns=['class']), train_df['class']
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=self.random_seed)
        # Get positions of train and val data in train_df
        train_positions, val_positions = next(sss.split(X, y))
        # Effectively perform the split while preserving initial indices
        self.train_df = train_df.iloc[train_positions]
        self.val_df = train_df.iloc[val_positions]

        return self.train_df, self.val_df
    
    def cv_splits(self, n_splits:int=5, shuffle:bool=True, kf=None):
        '''
        Generate cross-validation splits for the training data.
        Parameters:
        - n_splits (int, optional): Number of cross-validation splits. Default is 5.
        - shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.
        - kf (optional): Custom cross-validation splitter. If None, StratifiedKFold is used. Default is None.
        Returns:
        - Dict[int, Dict[str, pd.DataFrame]]: Dictionary containing cross-validation data splits.
        '''
        # If class has self.train_df, it means a split has already been done so we consider this, otherwise we consider the original df 
        train_df = self.train_df if hasattr(self, 'train_df') else self.original_df

        # Use kf if exists else use StratifiedKFold
        kfold = kf if kf else StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_seed)
            
        self.cv_indices = {}
        X, y = train_df.drop(columns=['class']), train_df['class']
            
        for i, (train_positions, val_positions) in enumerate(kfold.split(X, y)):
            self.cv_dfs[i]['train'] = train_df.iloc[train_positions]
            self.cv_dfs[i]['val'] = train_df.iloc[val_positions]

        return self.cv_dfs
    
    
    
class CustomDfImageFolder(Dataset):
    '''
    A custom Dataset class inheriting from torch.utils.data.Dataset to replicate the functionality of torchvision.datasets.ImageFolder.
    It uses a DataFrame containing image paths and class labels instead of directly reading from a directory.

    Parameters:
    - path_class_df (pd.DataFrame): DataFrame containing image paths and class labels.
    - transform (callable, optional): A function/transform to apply to the images. Default is None.
    - target_transform (callable, optional): A function/transform to apply to the labels. Default is None.
    
    Attributes:
    - path_class_df (pd.DataFrame): DataFrame containing image paths and corresponding class labels.
    - transform (callable, optional): A function/transform to apply to the images.
    - target_transform (callable, optional): A function/transform to apply to the labels.
    - classes (List[str]): Sorted list of class names.
    - class_to_idx (Dict[str, int]): Dictionary mapping class names to class indices.
    '''

    def __init__(self,
                 path_class_df: pd.DataFrame,
                 transform = None,
                 target_transform = None):
        
        self.path_class_df = path_class_df
        #self.path_class_df.reset_index(inplace=True, drop=True)
        self.transform = transform
        self.target_transform = target_transform
        
        # Run get_classes method during initialization to get self.classes and self.classes_dict
        self.classes, self.class_to_idx = self._get_classes()

    def _get_classes(self) -> Tuple[List[str], Dict[str, int]]:
        '''
        Get the class names and their corresponding indices.
        Returns: - Tuple[List[str], Dict[str, int]]: A tuple containing a list of class names and a dictionary mapping class names to indices.
        '''
        # Get the class names (unique values in label column)
        classes = sorted(self.path_class_df['class'].unique().tolist())
        # Get the dict of classes and associated labels
        class_to_idx = {this_class:label for label, this_class in enumerate(classes)}
        return classes, class_to_idx
    
    def __len__(self) -> int:
        '''
        Get the number of samples in the dataset.
        Returns: - int: Number of samples in the dataset.
        '''
        # Overwrite Dataset's __len__ method with the len of path_class_df (number of entries)
        return len(self.path_class_df)
    
    def _load_image(self, path: str):
        '''
        Load an image from the given path.
        Parameters: - path (str): Path to the image file.
        Returns: - Image.Image: Loaded image.
        '''
        return Image.open(path)
    
    def _convert_mode_L_to_RGB(self, image):
        '''
        Convert an image to RGB mode if it is not already in RGB mode.
        Parameters: - image (Image.Image): Image to be converted.
        Returns: - Image.Image: Image in RGB mode.
        '''
        # Convert to RGB if the mode is not already RGB
        return image.convert('RGB') if image.mode != 'RGB' else image
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        '''
        Get a sample from the dataset at the specified index.
        Parameters: - index (int): Index of the sample to retrieve.
        Returns: - Tuple[torch.Tensor, int]: A tuple containing the image tensor and the class label.
        '''
        # Overwrite Dataset's __getitem__ method to return one data sample (data, label) potentially transformed
        # Get image path and class via its position in the df (not its index since we preserved indexing of original df)
        img_path, img_class = self.path_class_df.iloc[index, 0], self.path_class_df.iloc[index, 1]
        # Load image
        image = self._load_image(img_path)
        image = self._convert_mode_L_to_RGB(image)
        # Get label
        class_label = self.class_to_idx[img_class]
        # Potentially transform image and label
        image = self.transform(image) if self.transform else image 
        class_label = self.target_transform(class_label) if self.target_transform else class_label
        return image, class_label
    
    
    
    
    
    
    
    
    
    

 