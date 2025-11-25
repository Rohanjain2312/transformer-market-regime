"""Transformer Model: Architecture for market regime classification"""

import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegimeClassifier(nn.Module):
    """
    Transformer model for market regime classification.
    
    Architecture:
        Input (batch, seq_len, n_features) 
        -> Linear projection to d_model
        -> Positional encoding
        -> Transformer encoder (n_layers)
        -> Global average pooling
        -> Classification head
        -> Output (batch, n_classes)
    """
    
    def __init__(self, n_features, d_model=None, n_heads=None, n_layers=None, 
                 d_ff=None, dropout=None, n_classes=None):
        super().__init__()
        
        # Use config defaults if not specified
        d_model = d_model or config.D_MODEL
        n_heads = n_heads or config.N_HEADS
        n_layers = n_layers or config.N_LAYERS
        d_ff = d_ff or config.D_FF
        dropout = dropout or config.DROPOUT
        n_classes = n_classes or config.NUM_CLASSES
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, n_features)
        
        Returns:
            logits: Tensor of shape (batch, n_classes)
        """
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, n_classes)
        
        return logits
    
    def get_num_parameters(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(n_features, device=None):
    """
    Create and initialize Transformer model.
    
    Args:
        n_features: Number of input features
        device: Device to move model to (cuda/cpu)
    
    Returns:
        model: Initialized Transformer model
        device: Device model is on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerRegimeClassifier(n_features=n_features)
    model = model.to(device)
    
    return model, device


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRANSFORMER MODEL TEST")
    print("="*70)
    
    # Model parameters
    n_features = len(config.BASELINE_FEATURES)
    batch_size = 32
    seq_len = config.SEQUENCE_LENGTH
    
    # Create model
    print("\nInitializing model...")
    model, device = create_model(n_features)
    
    # Model summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    print(f"Device:                {device}")
    print(f"Input Features:        {n_features}")
    print(f"Model Dimension:       {config.D_MODEL}")
    print(f"Attention Heads:       {config.N_HEADS}")
    print(f"Encoder Layers:        {config.N_LAYERS}")
    print(f"Feedforward Dim:       {config.D_FF}")
    print(f"Dropout:               {config.DROPOUT}")
    print(f"Output Classes:        {config.NUM_CLASSES}")
    print(f"Total Parameters:      {model.get_num_parameters():,}")
    
    # Test forward pass
    print("\n" + "="*70)
    print("FORWARD PASS TEST")
    print("="*70)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, n_features).to(device)
    print(f"Input shape:           {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape:          {output.shape}")
    print(f"Output dtype:          {output.dtype}")
    
    # Check output properties
    probs = torch.softmax(output, dim=1)
    print(f"Probability sum:       {probs.sum(dim=1)[0]:.4f} (should be ~1.0)")
    print(f"Predictions:           {output.argmax(dim=1)[:5].tolist()}")
    
    print("\nValidation:            PASS")
    
    print("\n" + "="*70)
    print("MODEL TEST COMPLETE")
    print("="*70 + "\n")