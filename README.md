# pytorch_vision_framework

## Project description:
This repository aims to provides a comprehensive and flexible framework for managing deep learning workflows with extensive configuration management. It addresses common challenges in dataset preparation, configuration management, and model evaluation by offering modular and extensible solutions. The framework is designed to streamline the process of loading data, applying transformations, training models, and evaluating their performance with a few lines of Python code.
All data processing parameters, model hyperparameters and more are easily set up through a .yml config file (template located in the conf folder of the repo) to adjust your workflow without changing the code.
For finetuning, please have a look to this repo: https://github.com/ValentinOzeel/optuna_tuning_through_config

### Usecase demonstration (Brain tumor classification based on MRI): 
Have a look to this use case exemple covering how to use the framework:
https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification

### Description of some features:
- Extensive configuration via .yml file (template located in the conf folder of the repo) is handled by the ConfigLoad class in the secondary_module.py.
- The LoadOurData class (data_loading.py module), is designed to handle data loading, preprocessing, and managing datasets and dataloaders for deep learning tasks using PyTorch. The class supports functionalities for training, validation, testing, and cross-validation datasets.
- The classes present in the model.py module combine best practices in deep learning model implementation and provides a comprehensive framework for training, evaluating pytorch models for vision tasks mainly. 
- The workflow_class.py module defines a comprehensive DeepLearningVisionWorkflow class designed for orchestrating the end-to-end workflow of a deep learning project, particularly in the domain of image processing using PyTorch. It enables to orchestrate the entire deep learning workflow, ensuring that users can efficiently train, evaluate, and infer with their models while maintaining flexibility and reproducibility.

### Installation:
- Install poetry
https://python-poetry.org/docs/

- Clone the repository

        git clone https://github.com/ValentinOzeel/pytorch_vision_framework.git

- cd to the corresponding folder

        cd Your/Path/To/The/Cloned/Repo  

- Activate your virtual environment with your favorite environment manager such as venv or conda (or poetry will create one)

- Run the installation process:

        poetry install

- Install or reinstall torch if GPU acceleration needed
https://pytorch.org/get-started/locally/
