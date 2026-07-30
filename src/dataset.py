import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TrafficDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int, pred_length: int):
        """
        PyTorch Dataset for Time-Series Traffic Data.
        
        Args:
            data (np.ndarray): Scaled traffic data.
            seq_length (int): Number of historical time steps to use as input.
            pred_length (int): Number of future time steps to predict.
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
        
    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length : index + self.seq_length + self.pred_length]
        
        # Convert to PyTorch tensors
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloaders(csv_file: str, seq_length: int = 12, pred_length: int = 3, batch_size: int = 32, train_split: float = 0.8):
    """
    Reads data, applies scaling, and returns PyTorch DataLoaders.
    
    Args:
        csv_file (str): Path to the traffic data CSV.
        seq_length (int): History window size.
        pred_length (int): Prediction window size.
        batch_size (int): Batch size for training.
        train_split (float): Ratio of data to use for training.
        
    Returns:
        train_loader, test_loader, scaler
    """
    df = pd.read_csv(csv_file)
    
    # Ensure data is sorted chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # We will use 'speed' as our target feature
    values = df[['speed']].values
    
    # Normalize the data (important for neural networks)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    # Split into train and test sets
    train_size = int(len(scaled_values) * train_split)
    train_data = scaled_values[:train_size]
    test_data = scaled_values[train_size:]
    
    # Create Dataset objects
    train_dataset = TrafficDataset(train_data, seq_length, pred_length)
    test_dataset = TrafficDataset(test_data, seq_length, pred_length)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader, scaler
