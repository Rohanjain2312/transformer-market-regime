"""Transformer Models: Multiple architectures for market regime classification"""

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
    Vanilla Transformer (Model 0 & 1)
    Architecture: Input → Linear → Positional → Transformer → Global Avg Pool → Classifier
    """
    
    def __init__(self, model_config):
        super().__init__()
        
        n_features = model_config['n_features']
        d_model = model_config['d_model']
        n_heads = model_config['n_heads']
        n_layers = model_config['n_layers']
        d_ff = model_config['d_ff']
        dropout = model_config['dropout']
        
        self.d_model = d_model
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, config.NUM_CLASSES)
        )
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionPoolingTransformer(nn.Module):
    """
    Model 3: Attention-Weighted Pooling + Residual Connections
    Architecture: Input → Transformer → Attention Pool → Residual → Classifier
    """
    
    def __init__(self, model_config):
        super().__init__()
        
        n_features = model_config['n_features']
        d_model = model_config['d_model']
        n_heads = model_config['n_heads']
        n_layers = model_config['n_layers']
        d_ff = model_config['d_ff']
        dropout = model_config['dropout']
        
        self.d_model = d_model
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Attention pooling
        self.attention_weights = nn.Linear(d_model, 1)
        
        # Residual projection
        self.residual_proj = nn.Linear(n_features, d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, config.NUM_CLASSES)
        )
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Store input for residual
        x_input = x
        
        # Transformer encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x_encoded = self.transformer_encoder(x)
        
        # Attention pooling
        attn_scores = self.attention_weights(x_encoded)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        x_pooled = (x_encoded * attn_weights).sum(dim=1)  # (batch, d_model)
        
        # Residual connection from input
        x_residual = self.residual_proj(x_input.mean(dim=1))
        x_final = x_pooled + x_residual
        
        return self.classifier(x_final)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNTransformer(nn.Module):
    """
    Model 4: CNN + Transformer Hybrid
    Architecture: Input → 1D CNN → Transformer → Classifier
    """
    
    def __init__(self, model_config):
        super().__init__()
        
        n_features = model_config['n_features']
        d_model = model_config['d_model']
        n_heads = model_config['n_heads']
        n_layers = model_config['n_layers']
        d_ff = model_config['d_ff']
        dropout = model_config['dropout']
        cnn_channels = model_config.get('cnn_channels', 64)
        cnn_kernel = model_config.get('cnn_kernel', 3)
        
        # CNN layers
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.conv2 = nn.Conv1d(cnn_channels, d_model, kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, config.NUM_CLASSES)
        )
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)  # Back to (batch, seq_len, d_model)
        
        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        
        return self.classifier(x)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiScaleTransformer(nn.Module):
    """
    Model 5: Multi-Scale Transformer
    Architecture: Parallel transformers on different window sizes → Concatenate → Classifier
    """
    
    def __init__(self, model_config):
        super().__init__()
        
        n_features = model_config['n_features']
        d_model = model_config['d_model']
        n_heads = model_config['n_heads']
        n_layers = model_config['n_layers']
        d_ff = model_config['d_ff']
        dropout = model_config['dropout']
        windows = model_config.get('windows', [30, 60, 90])
        
        self.windows = windows
        self.n_scales = len(windows)
        
        # Create separate transformer for each scale
        self.transformers = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.pos_encoders = nn.ModuleList()
        
        for _ in windows:
            # Input projection
            self.projections.append(nn.Linear(n_features, d_model))
            self.pos_encoders.append(PositionalEncoding(d_model))
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True
            )
            self.transformers.append(nn.TransformerEncoder(encoder_layer, num_layers=n_layers))
        
        # Classifier on concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(d_model * self.n_scales, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, config.NUM_CLASSES)
        )
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: (batch, 60, features)
        outputs = []
        
        for i, window_size in enumerate(self.windows):
            # Extract last window_size timesteps
            if x.size(1) >= window_size:
                x_window = x[:, -window_size:, :]
            else:
                # Pad if sequence shorter than window
                x_window = x
            
            # Process through transformer
            x_proj = self.projections[i](x_window)
            x_pos = self.pos_encoders[i](x_proj)
            x_encoded = self.transformers[i](x_pos)
            x_pooled = x_encoded.mean(dim=1)
            outputs.append(x_pooled)
        
        # Concatenate all scales
        x_multi = torch.cat(outputs, dim=1)
        return self.classifier(x_multi)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMTransformer(nn.Module):
    """
    Model 6: LSTM + Transformer Hybrid
    Architecture: Input → LSTM → Transformer → Classifier
    """
    
    def __init__(self, model_config):
        super().__init__()
        
        n_features = model_config['n_features']
        d_model = model_config['d_model']
        n_heads = model_config['n_heads']
        n_layers = model_config['n_layers']
        d_ff = model_config['d_ff']
        dropout = model_config['dropout']
        lstm_hidden = model_config.get('lstm_hidden', 64)
        lstm_layers = model_config.get('lstm_layers', 2)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Project LSTM output to d_model
        self.lstm_proj = nn.Linear(lstm_hidden, d_model)
        
        # Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, config.NUM_CLASSES)
        )
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = self.lstm_proj(lstm_out)
        
        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        
        return self.classifier(x)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_id, device=None):
    """
    Factory function to create model based on ID
    
    Args:
        model_id: Model identifier (e.g., 'model_0_baseline', 'model_2_large_capacity')
        device: Device to move model to
    
    Returns:
        model: Initialized model
        device: Device model is on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = config.get_model_config(model_id)
    architecture = model_config['architecture']
    
    # Select architecture
    if architecture == 'vanilla':
        model = TransformerRegimeClassifier(model_config)
    elif architecture == 'attention_pooling':
        model = AttentionPoolingTransformer(model_config)
    elif architecture == 'cnn_transformer':
        model = CNNTransformer(model_config)
    elif architecture == 'multiscale':
        model = MultiScaleTransformer(model_config)
    elif architecture == 'lstm_transformer':
        model = LSTMTransformer(model_config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model = model.to(device)
    return model, device


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE TEST")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    seq_len = config.SEQUENCE_LENGTH
    
    # Test all models
    for model_id in config.list_available_models():
        print(f"\n{model_id}:")
        print("-"*70)
        
        model_config = config.get_model_config(model_id)
        model, _ = create_model(model_id, device)
        
        n_features = model_config['n_features']
        dummy_input = torch.randn(batch_size, seq_len, n_features).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Name:           {model_config['name']}")
        print(f"Architecture:   {model_config['architecture']}")
        print(f"Input shape:    {dummy_input.shape}")
        print(f"Output shape:   {output.shape}")
        print(f"Parameters:     {model.get_num_parameters():,}")
    
    print("\n" + "="*70)
    print("ALL MODELS TESTED SUCCESSFULLY")
    print("="*70 + "\n")