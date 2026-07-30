import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, num_layers=2, dim_feedforward=256, dropout=0.1, pred_length=3):
        """
        Transformer model for time series forecasting.
        
        Args:
            input_dim (int): Number of input features (1 for just speed).
            d_model (int): Dimension of the model embeddings.
            n_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network model.
            dropout (float): Dropout probability.
            pred_length (int): Number of future steps to predict.
        """
        super(TimeSeriesTransformer, self).__init__()
        
        # Linear layer to map input sequence to d_model dimensions
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer mapping the combined representations to the prediction horizon
        # We flatten the sequence and predict the future steps
        self.flatten = nn.Flatten()
        
        # We will determine the linear input size dynamically in forward pass or we can pass seq_length
        # For simplicity, we pool the temporal dimension (e.g. mean or take the last representation)
        self.output_linear = nn.Linear(d_model, pred_length)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
        Returns:
            output: Tensor, shape [batch_size, pred_length]
        """
        # Map input to d_model
        x = self.input_linear(src) # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x) # [batch_size, seq_len, d_model]
        
        # Take the representation of the last time step for prediction
        # Alternatively, we could use average pooling: x = x.mean(dim=1)
        x_last = x[:, -1, :] # [batch_size, d_model]
        
        # Final linear layer to output predictions
        output = self.output_linear(x_last) # [batch_size, pred_length]
        
        # Add a feature dimension to match target shape [batch_size, pred_length, 1]
        return output.unsqueeze(-1)
