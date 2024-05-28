import os
import random 
from PIL import Image
from typing import List
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt


class DataExploration():
    def __init__(self, 
                 project_root_path:str, 
                 RANDOM_SEED=None, 
                 transformation_steps=None):
        
        self.project_root_path = project_root_path
        self.RANDOM_SEED = RANDOM_SEED
        random.seed(self.RANDOM_SEED)
        self.transformation_steps = transformation_steps
        
    def set_random_seed(self, RANDOM_SEED):
        self.RANDOM_SEED = RANDOM_SEED
        random.seed(self.RANDOM_SEED)
        return
        
    def walk_through_dir(self, dir_path=None):
        if dir_path is None:
            dir_path = self.project_root_path
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"In {dir_path}, there are:\n- {len(dirnames)} Directories\n- {len(filenames)} Files")
        return
        
    def get_all_image_paths(self, data_path:Path = None) -> List:
        if data_path is None:
            data_path = Path(os.path.join(self.project_root_path, 'data'))
        return list(data_path.glob("*/*/*.jpg"))

    def get_a_random_path_class_image(self):
        # Get all image (train/test) paths
        all_image_paths = self.get_all_image_paths()
        # Pick a random image path
        random_image_path = random.choice(all_image_paths)
        # Get image class (name of parent directory)
        image_class = random_image_path.parent.stem
        # Open the random image
        random_img = Image.open(random_image_path)
        return random_image_path, image_class, random_img


    def get_random_images_w_class(self, n=3):
        # Get all image (train/test) paths
        all_image_paths = self.get_all_image_paths()
        random_paths = random.sample(all_image_paths, k=n)
        return [(Image.open(path), path.parent.stem) for path in random_paths]

    def define_transformation_steps(self, transformation_steps: List):
        self.transformation_steps = transformation_steps
        
    def transform_data_w_torchvision(self, data: List, custom_transform=None):
        # If data is a single image, convert it to a list with one element
        if not isinstance(data, list):
            data = [data]

        # Use custom_transform if not None else use defined steps
        transformation_steps = custom_transform if custom_transform else self.transformation_steps
        if not transformation_steps:
            raise ValueError('No transformation steps available.')
        
        # Create compose transform
        composed_transform = transforms.Compose(transformation_steps)
        # Apply the composed transformation to each image in the list
        transformed_data = [composed_transform(img) for img in data]
        # If there was only one image initially, return the transformed image directly
        return transformed_data[0] if len(transformed_data) == 1 else transformed_data

    
    def plot_transformed_images(self, custom_transform=None, n=5):
        '''
        Select random images from list of path, transform them
        '''
        # Get list of tuples [(image, class), ...]
        images_w_class = self.get_random_images_w_class(n=n)
        # Unpack the list of tuples 
        images, image_classes = [list(item) for item in zip(*images_w_class)]
    
        # Get transformed images
        if custom_transform:
            transformed_images = self.transform_data_w_torchvision(images, custom_transform=custom_transform)
        else:
            transformed_images = self.transform_data_w_torchvision(images)
        
        # Create subpolts of 1 lines and 2 cols for each image
        fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(2*n, 2.5*n))
        fig.suptitle("Plot of original/transformed images", fontsize=20)
        # Loop over images, classes and transformed images for plotting
        for i, (image, transformed_image, image_class) in enumerate(zip(images, transformed_images, image_classes)):
            
            # Move color channel to the last dimension (Color, Height, Width -> Height, Width, Color)
            transformed_image = transformed_image.permute(1, 2, 0)
            
            # Original image
            ax[i, 0].imshow(image)
            ax[i, 0].set_title(f"Initial image (class: {image_class})\nSize: {image.size}")
            ax[i, 0].axis(False)
            
            #Transformed images
            ax[i, 1].imshow(transformed_image)
            ax[i, 1].set_title(f"Transformed image (class: {image_class})\nShape: {transformed_image.shape}")
            ax[i, 1].axis(False)        
            

        plt.tight_layout()
        plt.show()
        
        

if __name__ == "__main__":
    # Assuming data_exploration.py is in src\data_exploration.py
    data_path = input('Please provide your data folder path.')
    if os.path.isdir(data_path):
        data_path = Path(data_path)

    # Initiate instance of the DataExploration class
    explore_data = DataExploration()
    # Walk through our data directory
    explore_data.walk_through_dir(data_path)
    # Get random image
    random_image_path, image_class, random_img = explore_data.get_a_random_path_class_image()
    # Print random image's metadata
    print('\nChoosing a random image to print its metadata...')
    print(f"Random image's path: {random_image_path}")
    print(f"Random image's class: {image_class}")
    print(f"Random image's height and width: {random_img.height} / {random_img.width}\n")
    
    # Define tranformation steps
    transform_steps = [
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandAugment(num_ops=3, magnitude=12),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()
    ]
    explore_data.define_transformation_steps(transformation_steps=transform_steps)
    # Check if image is transformed (should be a tensor)
    transformed_data = explore_data.transform_data_w_torchvision(random_img)
    print(transformed_data.shape)
    # Plot original vs transformed images
    explore_data.plot_transformed_images()