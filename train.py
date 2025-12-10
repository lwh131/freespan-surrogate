import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np
import os
import pandas as pd
from early_stopping import EarlyStopping
from models import init_weights_xavier, InceptCurvesFiLM
from evaluation import evaluate
from preprocessing import DataScaler


class WeightedMSELoss(nn.Module):
    """
    Calculates a weighted Mean Squared Error, allowing different components of the
    output to be penalized differently.
    """
    def __init__(self, weights):
        """
        Args:
            weights (torch.Tensor): A 1D tensor of weights, one for each output feature.
                                    The weights should be on the same device as the model.
        """
        super().__init__()
        # register_buffer ensures the weights tensor is moved to the correct device
        # (e.g., GPU) when model.to(device) is called, but it is not considered
        # a trainable model parameter.
        self.register_buffer('weights', weights)

    def forward(self, predictions, targets):
        """
        Calculates the weighted loss.
        Args:
            predictions (torch.Tensor): The model's output. Shape: [batch_size, num_outputs]
            targets (torch.Tensor): The ground truth values. Shape: [batch_size, num_outputs]
        """
        # Calculate the squared error for each element
        # Shape: [batch_size, num_outputs]
        squared_errors = (predictions - targets) ** 2
        
        # Apply the weights. The weights tensor of shape [num_outputs] will be
        # broadcasted to match the shape of the squared_errors tensor.
        weighted_squared_errors = squared_errors * self.weights
        
        # Return the mean of all weighted errors
        return torch.mean(weighted_squared_errors)


def prepare_data(window_path, seabed_path, scalar_path, targets_path):
    '''
    Loads training data.
    '''

    df_windows = pd.read_parquet(window_path, engine='pyarrow')

    splits = df_windows['split'].to_numpy()

    # Cable properties
    scalar_inputs_array = df_windows[['EA', 'EI', 'submerged_weight', 'residual_lay_tension']].to_numpy()
    scalar_scaler = DataScaler()
    scalar_scaler.fit(scalar_inputs_array[splits == 'training'])        # Fit scaler only on training data
    scaled_scalars_tensor = torch.from_numpy(scalar_scaler.transform(scalar_inputs_array)).float()
    scalar_scaler.save(scalar_path)

    # Seabed profile
    seabed_z_profile_array = np.stack(df_windows['window_zs'].tolist())
    profile_scaler = DataScaler()
    profile_scaler.fit(seabed_z_profile_array[splits == 'training'])
    seabed_profile_tensor = torch.from_numpy(profile_scaler.transform(seabed_z_profile_array)).float()
    profile_scaler.save(seabed_path)

    # Slopes
    targets = df_windows[['centre_start_z', 'centre_end_z', 'fea_slope_start', 'fea_slope_end']].to_numpy()
    # targets = df_windows[['centre_start_z', 'centre_end_z']].to_numpy()
    target_scaler = DataScaler()
    target_scaler.fit(targets[splits == 'training'])
    scaled_target_tensor = torch.from_numpy(target_scaler.transform(targets)).float()
    target_scaler.save(targets_path)

    # Assemble
    dataset = TensorDataset(scaled_scalars_tensor, seabed_profile_tensor, scaled_target_tensor)

    training_indices = np.where(splits == 'training')[0]
    validation_indices = np.where(splits == 'validation')[0]
    
    train_dataset = torch.utils.data.Subset(dataset, training_indices)
    val_dataset = torch.utils.data.Subset(dataset, validation_indices)

    return train_dataset, val_dataset, scalar_scaler, profile_scaler, target_scaler


def train(model, optimiser, scheduler, train_loader, validation_loader, epochs, device, early_stopper, model_params, criterion):
    start_time = time.time()

    train_losses = []
    validation_losses = []
    
    base_filename = 'learning_curve_windows'
    i = 0
    while os.path.exists(f'{base_filename}_{i}.png'):
        i += 1
    plot_filename = f'{base_filename}_{i}.png'
    print(f"Saving learning curve for this run to: {plot_filename}")

    print("Starting Training")
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        epoch_training_loss = 0
        for scalar_features, seabed_profiles, targets in train_loader:
            scalar_features = scalar_features.to(device)
            seabed_profiles = seabed_profiles.to(device)
            targets = targets.to(device)
            
            # print(f"seabed_input shape: {seabed_profiles.shape}")
            # print(f"scalar_input shape: {scalar_features.shape}")
            predictions = model(seabed_profiles, scalar_features)
            
            loss = criterion(predictions, targets)
                
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimiser.step()
            
            epoch_training_loss += loss.item()

        avg_train_loss = epoch_training_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase 
        model.eval()
        epoch_validation_loss = 0
        with torch.no_grad():
            for scalar_features, seabed_profiles, targets in validation_loader:
                scalar_features = scalar_features.to(device)
                seabed_profiles = seabed_profiles.to(device)
                targets = targets.to(device)
                predictions = model(seabed_profiles, scalar_features)
                loss = criterion(predictions, targets)
                epoch_validation_loss += loss.item()
        avg_validation_loss = epoch_validation_loss / len(validation_loader)
        validation_losses.append(avg_validation_loss)

        # Print status and update plot
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1:>{len(str(epochs))}}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_validation_loss:.6f} | LR: {current_lr:.6f}')
            
            # Visualization
            plt.style.use('bmh')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_losses, label='Training Loss', color='blue')
            ax.plot(validation_losses, label='Validation Loss', color='orange')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('Training and Validation Loss vs. Epochs')
            ax.legend()
            ax.grid(True)
            plt.yscale('log')
            
            # Overwrite the same file on each update
            plt.savefig(plot_filename, dpi=300)
            plt.close(fig) # Close the figure to free up memory

        early_stopper(avg_validation_loss, model, model_params)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
        scheduler.step(avg_validation_loss)
        
    end_time = time.time()
    print(" Training Complete ")

    return start_time, end_time


#Hyperparameters
# TRAINING
EPOCHS = 10000
BATCH_SIZE = 512
STOPPING_PATIENCE = 200
# OPTIMISER
LEARN_RATE = 1e-5
# REGULARISATION_WEIGHT_DECAY = 1e-3
# SCHEDULER
SCHEDULER_PATIENCE = 150
SCHEDULER_FACTOR = 0.5
# LOSS FUNCTION
Z_WEIGHT = 1.0
SLOPE_WEIGHT = 1.0
#Files
MASTER_FILE_PATH = 'master_batch_file.parquet'
SPAN_FILE_PATH = 'master_windows_file.parquet'
SEABED_SCALER_PATH = 'seabed_profile_scaler.joblib'
FEATURE_SCALER_PATH = 'feature_scaler.joblib'
TARGETS_SCALER_PATH = 'target_scaler.joblib'
MODEL_PATH = 'window_solver.pth'

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

loss_weights = torch.tensor([
    Z_WEIGHT,      # Weight for centre_start_z
    Z_WEIGHT,      # Weight for centre_end_z
    SLOPE_WEIGHT,  # Weight for fea_slope_start
    SLOPE_WEIGHT   # Weight for fea_slope_end
]).to(device)

criterion = WeightedMSELoss(weights=loss_weights)

# Prepare data
train_dataset, val_dataset, scalar_scaler, profile_scaler, target_scaler = prepare_data(SPAN_FILE_PATH, SEABED_SCALER_PATH, FEATURE_SCALER_PATH, TARGETS_SCALER_PATH)

training_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_SCALAR_PARAMETERS = len(scalar_scaler.scaler.mean_)
NUM_TARGET_PARAMETERS = len(target_scaler.scaler.mean_)
model_params = {
    'num_scalars': NUM_SCALAR_PARAMETERS,
    'num_outputs': NUM_TARGET_PARAMETERS
}
model = InceptCurvesFiLM(**model_params).to(device).apply(init_weights_xavier)

optimiser = optim.Adam(model.parameters(), lr=LEARN_RATE)
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
early_stopper = EarlyStopping(patience=STOPPING_PATIENCE, verbose=True, path=MODEL_PATH)

print("Starting training InceptCurvesFiLM")

train(model, optimiser, scheduler, training_loader, validation_loader, EPOCHS, device, early_stopper, model_params, criterion)

evaluate(num_samples=5, windows=True)