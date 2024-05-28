import os
import numpy as np
import copy
from typing import Tuple, Dict, List
from tqdm.auto import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader

import gzip
import shutil

from .secondary_module import colorize

import matplotlib.pyplot as plt


class MRI_CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) architecture for MRI tumors classification.

    Args:
        input_shape (Tuple[int]): The shape of the input data. Should be in the format 
            (n_images, color_channels, height, width).
        hidden_units (int): The number of hidden units to use in the convolutional layers.
        output_shape (int): The number of classes for classification.
        activation_func (torch.nn.Module): The activation function to use after each convolutional layer.

    Attributes:
        input_shape (Tuple[int]): The shape of the input data.
        hidden_units (int): The number of hidden units in the convolutional layers.
        output_shape (int): The number of classes for classification.
        activation_func (torch.nn.Module): The activation function used after each convolutional layer.
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        conv3 (nn.Conv2d): The third convolutional layer.
        pool (nn.MaxPool2d): The max pooling layer.
        conv_layers (nn.Sequential): Sequential container for the convolutional layers.
        last_conv_output_shape (torch.Size): The output shape of the last convolutional layer.
        flatten (nn.Flatten): Flatten layer to flatten the output of the last convolutional layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer for classification.
        dropout (nn.Dropout): Dropout layer for regularization.
        dense_layers (nn.Sequential): Sequential container for the fully connected layers.

    Methods:
        forward(x): Performs forward pass through the network.

    """

    def __init__(self,                  
                 input_shape:Tuple[int],
                 hidden_units:int,
                 output_shape:int,
                 activation_func:torch.nn
                 ):

        self.input_shape = input_shape # [n_images, color_channels, height, width]
        self.hidden_units = hidden_units
        self.output_shape = output_shape # Number of classes
        self.activation_func = activation_func
        
        super(MRI_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[1], out_channels=self.hidden_units, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_units, out_channels=self._hidden_units(2), kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.hidden_units, out_channels=self._hidden_units(2), kernel_size=3, padding=1)
        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Chain convolutional layers
        self.conv_layers = nn.Sequential(
            self.conv1, self.activation_func(), self.pool,
            self.conv2, self.activation_func(), self.pool,
            self.conv3, self.activation_func(), self.pool
        )

        # Calculate the number of input features for the linear layer dynamically
        self.last_conv_output_shape = self._calculate_last_conv_layer_output_shape()
        
        # Flatten 
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_units * self.last_conv_output_shape[-2] * self.last_conv_output_shape[-1], self._hidden_units(4))
        self.fc2 = nn.Linear(self.hidden_units, self.output_shape)
        # Dropout regularization
        self.dropout = nn.Dropout(p=0.5)
        # Chain dense layers
        self.dense_layers = nn.Sequential(
            self.fc1, self.activation_func(), self.dropout,
            self.fc2
        )


    def _hidden_units(self, multiplier:int):
        """
        Helper method to update the number of hidden units.
        Args: multiplier (int): The multiplier to adjust the number of hidden units.
        Returns: int: The updated number of hidden units.
        """
        if isinstance(multiplier, int):
            self.hidden_units = self.hidden_units * multiplier
            return self.hidden_units
        
    def _calculate_last_conv_layer_output_shape(self):
        """
        Helper method to calculate the output shape of the last convolutional layer.
        Returns: torch.Size: The output shape of the last convolutional layer.
        """
        
        # Assuming input_shape is in the format (channels, height, width)
        dummy_input_output = torch.randn(*self.input_shape)
        with torch.inference_mode():
            # Pass dummy in all layers except for the last one
            for layer in self.conv_layers:
                dummy_input_output = layer(dummy_input_output)
        return dummy_input_output.shape
        
    def forward(self, x):
        return self.dense_layers(self.flatten(self.conv_layers(x)))
         

       
        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after last time validation loss improved. Default: 7
        delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        project_root (str): Path of the project root
        save_checkpoint (bool): Whether to save the checkpoint or not. Default: True
        saving_option (str): Option for saving checkpoint ('model', 'state_dict', 'onnx'). Default: 'model'
        save_dir_path (str): Path relative to the root to save checkpoint. Default: None
        compress (bool): Compress the model with gzip and shutil
        trace_func (function): Function to use for printing. Default: print
        verbose (bool): If True, prints a message for each validation loss improvement. Default: False
    """

    def __init__(self, 
                 patience=7, delta=0, 
                 project_root=None, save_checkpoint=True, saving_option='model', save_dir_path='', compress=False,
                 trace_func=print, verbose=False):

        self.patience = patience
        self.delta = delta
        self.save_checkpoint = save_checkpoint
        self.saving_option = saving_option.lower()
        self.save_dir_path = os.path.join(project_root, save_dir_path)
        self.compress = compress
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model_checkpoint = None
        self.best_epoch = None
        
        if self.save_checkpoint and not os.path.exists(self.save_dir_path):
            os.makedirs(self.save_dir_path)
            print(f'Model checkpoints will be saved at {self.save_dir_path}')
        
        if self.saving_option not in ['model', 'state_dict', 'onnx']:
            raise ValueError("EarlyStopping class' saving_option parameter should be either 'model', 'state_dict' or 'onnx'.")
        
    def __call__(self, epoch, val_loss, model, input_data_onnx=None):
        """Return self.early_stop value: True or False. Potentially save checkpoints.

        Args:
            val_loss (float): Validation loss value.
            model (torch.nn.Module): PyTorch model to save.
            input_data_onnx (Tensor or Tuple of Tensors): Needed for data signature when using ONNX. Default: None

        Returns:
            bool: Whether to early stop or not.
        """     
                            
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(epoch, val_loss, model, input_data_onnx)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(epoch, val_loss, model, input_data_onnx)
            self.counter = 0
            
        return self.early_stop

    def _save_checkpoint(self, epoch, val_loss, model, input_data_onnx):
        """Saves model when validation loss decrease.

        Args:
            val_loss (float): Validation loss value.
            model (torch.nn.Module): PyTorch model to save.
            input_data_onnx (Tensor or Tuple of Tensors): Needed for data signature when using ONNX.
        """
                
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased compared to the lowest value recorded ({self.val_loss_min:.6f} --> {val_loss:.6f}).\n')
        
        self.val_loss_min = val_loss
        self.best_model_checkpoint = model
        self.best_epoch = epoch
            
        if self.save_checkpoint:
            if self.saving_option == 'onnx' and not input_data_onnx:
                raise ValueError("kwargs[0]/input_data_onnx parameter should be assigned when calling early_stopping while using saving_option = 'onnx'.")
        
            if self.saving_option == 'model':
                save_path = os.path.join(self.save_dir_path, 'best_model_checkpoint.pt')
                torch.save(model, save_path)
                if self.compress: self.__compress(save_path)
            elif self.saving_option == 'state_dict':
                save_path = os.path.join(self.save_dir_path, 'best_model_state_checkpoint.pt')
                torch.save(model.state_dict(), save_path)
                if self.compress: self.__compress(save_path)
            elif self.saving_option == 'onnx':
                save_path = os.path.join(self.save_dir_path, 'best_model_checkpoint.onnx')
                torch.onnx.export(model, input_data_onnx, save_path)
                if self.compress: self.__compress(save_path)

    def __compress(self, model_path):
        # Compress the model file
        with open(model_path, 'rb') as f_in:
            with gzip.open(f'{model_path}.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # remove the original model file to save space
        os.remove(model_path)

class MetricsTracker:
    def __init__(self, metrics:List[str], n_classes:int, average:str='macro', torchmetrics:Dict={}):
        """
        Class for tracking and computing evaluation metrics.

        Args:
            metrics (List[str]): List of metric names to track. Available options are ['accuracy', 'precision', 'recall', 'f1'].
            n_classes (int): Number of classes in the classification problem.
            average (str, optional): Type of averaging to use for 'precision' and 'recall' metrics. Default is 'macro'.
            torchmetrics (Dict, optional): Dictionary containing TorchMetrics objects to track and compute. Default is an empty dictionary.

        Raises:
            ValueError: If an invalid average parameter is provided or if an invalid metric is selected.

        Attributes:
            metrics (List[str]): List of metric names to track.
            n_classes (int): Number of classes in the classification problem.
            average (str): Type of averaging to use for 'precision' and 'recall' metrics.
            torchmetrics (Dict): Dictionary containing TorchMetrics objects.
            all_metrics (List[str]): Combined list of tracked metrics and TorchMetrics names.
            available_metrics (List[str]): List of available metric names.

        Methods:
            reset(): Reset all tracked metrics.
            update(predictions, labels): Update tracked metrics based on new predictions and labels.
            accuracy(): Compute accuracy metric.
            precision(): Compute precision metric.
            recall(): Compute recall metric.
            f1(): Compute F1 score metric.
            compute_metrics(): Compute all tracked metrics.

        """
    
        self.metrics = metrics
        self.n_classes = n_classes
        self.average = average.lower()
        self.torchmetrics = torchmetrics
        
        self.all_metrics = self.metrics + list(self.torchmetrics.keys())
        
        if self.average not in ['macro', 'micro']:
            raise ValueError("Invalid average parameter. Please use 'macro' or 'micro'.")
        
        self.available_metrics = ['accuracy', 'precision', 'recall', 'f1']
        if set(self.metrics) - set(self.available_metrics):
            raise ValueError(f"Invalid 'metrics' parameter. Please only select available metrics ({self.available_metrics}) or use torchmetrics parameter.")
        
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.tp = torch.zeros(self.n_classes) if self.n_classes else 0
        self.fp = torch.zeros(self.n_classes) if self.n_classes else 0
        self.fn = torch.zeros(self.n_classes) if self.n_classes else 0
        self.total_correct = 0
        self.total_samples = 0
        
        for _, metric_obj in self.torchmetrics.items():
            metric_obj.reset()

    def update(self, predictions, labels):
        """Update tracked metrics based on new predictions and labels."""
        
        if 'accuracy' in self.metrics:
            self.total_correct += torch.sum(predictions == labels).item()
            self.total_samples += len(labels)

        if any(metric for metric in ['precision', 'recall', 'f1'] if metric in self.metrics):
            for cls in range(self.n_classes):
                self.tp[cls] += torch.sum((predictions == cls) & (labels == cls)).item()
                self.fp[cls] += torch.sum((predictions == cls) & (labels != cls)).item()
                self.fn[cls] += torch.sum((predictions != cls) & (labels == cls)).item()
        
        for _, metric_obj in self.torchmetrics.items():
            metric_obj.update(predictions, labels)
            

    def accuracy(self):
        return self.total_correct / self.total_samples

    def precision(self):
        if self.average == 'macro':
            return torch.mean(self.tp / (self.tp + self.fp + 1e-8))
        elif self.average == 'micro':
            return torch.sum(self.tp) / (torch.sum(self.tp) + torch.sum(self.fp) + 1e-8)

    def recall(self):
        if self.average == 'macro':
            return torch.mean(self.tp / (self.tp + self.fn + 1e-8))
        elif self.average == 'micro':
            return torch.sum(self.tp) / (torch.sum(self.tp) + torch.sum(self.fn) + 1e-8)

    def f1(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec + 1e-8)

    def compute_metrics(self) -> Dict:
        """
        Compute all tracked metrics.

        Returns:
            dict: Dictionary containing computed metrics.
        """
        # Add metrics
        metrics_dict = {metric:getattr(self, metric)() for metric in self.metrics}
        # Add torchmetrics
        for metric_name, metric_obj in self.torchmetrics.items():
            metrics_dict[metric_name] = metric_obj
        return metrics_dict

        
class TrainTestEval():
    """
    Class for training, validation, and evaluation of PyTorch models as well as for performing inference.

    Args:
        model (nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        loss_func (nn.Module): The loss function for computing the training loss.
        metrics_tracker (MetricsTracker): An instance of MetricsTracker for tracking evaluation metrics.
        epochs (int, optional): Number of epochs for training. Defaults to 10.
        lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        early_stopping (EarlyStopping, optional): Early stopping criterion. Defaults to None.
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer=None,
                 loss_func: nn.Module=None,
                 metrics_tracker: MetricsTracker=None,
                 epochs: int = 10,
                 lr_scheduler: torch.optim.lr_scheduler=None,
                 early_stopping:EarlyStopping=None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 random_seed: int = None
                ):
        
        self.model = model 
        self.optimizer = optimizer
        self.loss_func = loss_func 
        self.train_metrics_tracker = copy.deepcopy(metrics_tracker)
        self.val_metrics_tracker = copy.deepcopy(metrics_tracker)
        self.epochs = epochs 
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.device = device 
        
        self.curve_metrics = copy.deepcopy(self.train_metrics_tracker.metrics)
        self.curve_metrics.insert(0, 'loss')
        self.torchmetrics = copy.deepcopy(list(self.train_metrics_tracker.torchmetrics.keys()))
        self.all_metrics = self.curve_metrics + self.torchmetrics
        
        # Put model on device
        self.model.to(self.device)
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
        
    def get_dummy_input(self, dataloader:DataLoader):
        imgs, labels = next(iter(dataloader))
        return imgs.to(self.device)
        
    def training_step(self, train_dataloader:DataLoader):
        """
        Perform a single training step.
        Args: train_dataloader (DataLoader): DataLoader for the training dataset.
        Returns: float: Average training loss for the epoch.
        """
        # Activate training mode
        self.model.train()
        # Setup training loss and accuracy
        train_loss = 0

        # Loop over dataloader batches
        for i, (imgs, labels) in enumerate(train_dataloader):
            # Data to device
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            # Forward pass
            train_pred_logit = self.model(imgs)
            # Calculate loss and add it to train_loss
            loss = self.loss_func(train_pred_logit, labels)
            train_loss += loss.item()
            # Optimizer zero grad
            self.optimizer.zero_grad()
            # Loss backpropagation
            loss.backward()
            # Optimizer step
            self.optimizer.step()
            # Predictions
            predicted_classes = torch.argmax(torch.softmax(train_pred_logit, dim=1), dim=1)
            # Update metrics
            self.train_metrics_tracker.update(predicted_classes, labels)

        # Average loss per batch
        train_loss = train_loss / len(train_dataloader)
        return train_loss


    def validation_step(self, val_dataloader:DataLoader):
        """
        Perform a single validation step.
        Args: val_dataloader (DataLoader): DataLoader for the validation dataset.
        Returns: float: Average validation loss for the epoch.
        """
        # Model in eval mode
        self.model.eval()
        # Setup valid loss and accuracy 
        val_loss = 0

        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, labels) in enumerate(val_dataloader):
                # Set data to device
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                # Forward pass
                val_pred_logit = self.model(imgs)
                # Calculate val loss
                loss = self.loss_func(val_pred_logit, labels)
                val_loss += loss.item()
                # Predictions
                predicted_classes = val_pred_logit.argmax(dim=1)
                # Update metrics
                self.val_metrics_tracker.update(predicted_classes, labels)
                
        # Average loss per batch
        val_loss = val_loss / len(val_dataloader)
        return val_loss
    
    def _schedule_lr(self, metric):
        """
        Adjust learning rate based on the validation metric.
        Args: metric: The validation metric for adjusting the learning rate.
        """
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(metric)
        else:
            self.lr_scheduler.step()
        print('Lr value: ', self.optimizer.param_groups[0]['lr'])

    def _organize_metrics_dict(self, gathered_metrics, train_metrics=None, val_metrics=None):
        """
        Organize metrics dictionary.
        Args:
            gathered_metrics: Dictionary to store gathered metrics.
            train_metrics: Metrics computed during training.
            val_metrics: Metrics computed during validation.
        Returns:
            dict: Organized metrics dictionary.
        """
        for metric in self.all_metrics:
            gathered_metrics['train'][metric].append(train_metrics[metric])
            gathered_metrics['val'][metric].append(val_metrics[metric])
            
        return gathered_metrics
            
        
    def training(self, train_dataloader:DataLoader, val_dataloader:DataLoader, verbose: bool = True, real_time_plot_metrics:bool = True, save_metric_plot = True) -> Tuple:
        """
        Perform model training.

        Args:
            train_dataloader (DataLoader): DataLoader for the training dataset.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            verbose (bool, optional): Whether to print training/validation metrics. Defaults to True.
            real_time_plot_metrics (bool, optional): Whether to plot metrics in real-time. Defaults to True.
            save_metric_plot (bool, optional): Whether to save the metrics plot. Defaults to True.

        Returns:
            Tuple: The trained model and gathered metrics.
        """
        
        # Empty dict to track metrics
        gathered_metrics = {'train':{}, 'val':{}}
        for state in gathered_metrics.keys():
            for metric_name in self.all_metrics:
                gathered_metrics[state][metric_name] = []

        # Initialize plot
        if real_time_plot_metrics:
            plt.figure(figsize=(len(self.curve_metrics)*4, 8))
            plt.ion()  # Turn on interactive mode for dynamic plotting
            
        # Loop through epochs 
        for epoch in tqdm(range(self.epochs)):
            # Reset metrics
            self.train_metrics_tracker.reset()
            self.val_metrics_tracker.reset()
            # Train and validation steps
            train_loss = self.training_step(train_dataloader)
            val_loss = self.validation_step(val_dataloader)
            # Actualize gathered_metrics
            train_metrics = self.train_metrics_tracker.compute_metrics()
            train_metrics['loss'] = train_loss
            val_metrics = self.val_metrics_tracker.compute_metrics()
            val_metrics['loss'] = val_loss
            # Keep track of computed metrics
            gathered_metrics = self._organize_metrics_dict(gathered_metrics, train_metrics=train_metrics, val_metrics=val_metrics)
            # Print training/validation info
            if verbose:
                print_train_metrics, print_val_metrics = '', ''

                for metric_name in self.curve_metrics:
                    print_train_metrics = ''.join([print_train_metrics, colorize(''.join([metric_name, ': ']), "RED"), f"{train_metrics[metric_name]:.4f}", colorize(" | ", "LIGHTMAGENTA_EX")])
       
                for metric_name in self.curve_metrics:
                    print_val_metrics = ''.join([print_val_metrics, colorize(''.join([metric_name, ': ']), "BLUE"), f"{val_metrics[metric_name]:.4f}", colorize(" | ", "LIGHTMAGENTA_EX")])
                # Print metrics at each epoch
                print(
                    colorize("\nEpoch: ", "LIGHTGREEN_EX"), epoch,
                    '\n-- Train metrics --', print_train_metrics,   
                    '\n--  Val metrics  --', print_val_metrics       
                )
                
            # Plot the metrics curves
            if real_time_plot_metrics:
                self.real_time_plot_metrics(gathered_metrics)
            # Adjust learning rate
            if self.lr_scheduler:
                self._schedule_lr(val_loss)
            # Check for early_stopping
            if self.early_stopping:
                # If early stop, get the best checkpoint model and break the training loop
                if self.early_stopping(epoch, val_loss, self.model, self.get_dummy_input(train_dataloader)):
                    checkpoint_epoch = self.early_stopping.best_epoch
                    self.model = self.early_stopping.best_model_checkpoint
                    break

        if save_metric_plot:
            plt.ioff()  # Turn off interactive mode
            plt.clf()
            self.save_plot_metrics(gathered_metrics, checkpoint_epoch=checkpoint_epoch if checkpoint_epoch else None) 
            self.save_plot_torchmetrics(gathered_metrics)
            
        return self.model, gathered_metrics
        
        
    def real_time_plot_metrics(self, gathered_metrics:Dict):  
        """
        Plot the training and validation metrics in real time after each epoch.
        Args: gathered_metrics (Dict): Dictionary containing the gathered metrics.
        """
        
        train_metrics, val_metrics = gathered_metrics['train'], gathered_metrics['val']
        for i, metric_name in enumerate(self.curve_metrics):
            plt.subplot(1, len(self.curve_metrics), i+1)
            plt.plot(range(len(train_metrics[metric_name])), train_metrics[metric_name], label=''.join(['train_', metric_name]), color='red')
            plt.plot(range(len(val_metrics[metric_name])), val_metrics[metric_name], label=''.join(['val_', metric_name]), color='blue')
            if not plt.gca().get_title(): 
                plt.title(f"train_{metric_name} VS val_{metric_name}")
                plt.xlabel('Epochs')
                plt.ylabel(metric_name)
                plt.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
        
    def save_plot_metrics(self, gathered_metrics:Dict, checkpoint_epoch:int=None):
        train_metrics, val_metrics = gathered_metrics['train'], gathered_metrics['val']
        for i, metric_name in enumerate(self.curve_metrics):
            plt.subplot(1, len(self.curve_metrics), i+1)
            plt.plot(range(len(train_metrics[metric_name])), train_metrics[metric_name], label=''.join(['train_', metric_name]), color='red')
            plt.plot(range(len(val_metrics[metric_name])), val_metrics[metric_name], label=''.join(['val_', metric_name]), color='blue')
            if checkpoint_epoch:
                # Add early stopping delimitation on graph
                plt.axvline(x=checkpoint_epoch, color='#260026', linestyle='--', linewidth=0.5, label='Last model checkpoint')
         
            plt.title(f"train_{metric_name} VS val_{metric_name}")
            plt.xlabel('Epochs')
            plt.ylabel(metric_name)
            plt.legend()
            
        plt.tight_layout()
        fig = plt.gcf()  # Get the current figure
        fig.savefig('training_metrics.png')
        plt.clf()
        
    def save_plot_torchmetrics(self, gathered_metrics:Dict):
        """
        Save the plots of torchmetrics.
        Args: gathered_metrics (Dict): Dictionary containing the gathered metrics.
        """
        for metric in self.torchmetrics:
            fig = plt.figure(figsize=(10, 6), layout="constrained")
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            gathered_metrics['train'][metric][-1].plot(ax=ax1)
            ax1.set_title(f"train_{metric}")
            gathered_metrics['val'][metric][-1].plot(ax=ax2)
            ax2.set_title(f"val_{metric}")
            fig.savefig(f'{metric}.png')
            plt.clf
        
    def cross_validation(self, cross_val_dataloaders:Dict) -> List:
        """
        Perform cross-validation.
        Args: cross_val_dataloaders (Dict): Dictionary containing DataLoaders for cross-validation.
        Returns: List: List of training metrics for each fold.
        """
        training_metrics_per_fold = []
        for fold, (train_dataloader, val_dataloader) in enumerate(zip(cross_val_dataloaders['train'], cross_val_dataloaders['val'])):
            training_metrics_per_fold.append(self.training(train_dataloader, val_dataloader, save_metric_plot=False))
        return training_metrics_per_fold
    
        
    def evaluation(self, test_dataloader:DataLoader) -> Tuple:
        """
        Perform model evaluation on the test dataset.
        Args: test_dataloader (DataLoader): DataLoader for the test dataset.
        Returns: Tuple: Test loss and accuracy.
        """
        # Model in eval mode
        self.model.eval()
        # Setup test loss and accuracy 
        test_loss, test_acc = 0, 0

        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, labels) in enumerate(test_dataloader):
                # Set data to device
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                # Forward pass
                test_pred_logit = self.model(imgs)
                # Calculate test loss
                loss = self.loss_func(test_pred_logit, labels)
                test_loss += loss.item()
                # Calculate accuracy
                predicted_classes = test_pred_logit.argmax(dim=1)
                test_acc += ((predicted_classes==labels).sum().item()/len(predicted_classes))

        # Average metrics per batch
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)
        
        print(
            colorize("\nModel evaluation: ", "LIGHTGREEN_EX"),
            colorize("test_loss: ", "RED"), f"{test_loss:.4f}", colorize(" | ", "LIGHTMAGENTA_EX"),
            colorize("test_acc: ", "RED"), f"{test_acc:.4f}", colorize(" | ", "LIGHTMAGENTA_EX"),
        )
                
        return test_loss, test_acc


    def inference(self, dataloader:DataLoader) -> List:
        """
        Perform inference on a given dataset.
        Args: dataloader (DataLoader): DataLoader for the dataset to infer.
        Returns: List: Predicted classes.
        """
        # Model in eval mode
        self.model.eval()

        pred_classes = []
        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, _) in enumerate(dataloader):
                # Set data to device
                imgs = imgs.to(self.device)
                # Forward pass
                pred_logit = self.model(imgs)
                # Get predicted classes
                predicted_classes = pred_logit.argmax(dim=1)
                # Extend predictions lists
                pred_classes.extend(predicted_classes.cpu().numpy().tolist())
   
        return pred_classes