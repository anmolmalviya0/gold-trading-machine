"""
TERMINAL - SIMPLIFIED LSTM Model (Defibrillator Protocol)
===========================================================
Simplified architecture to prevent sigmoid saturation.
Removed: Bidirectional, Attention, Deep FC layers
Added: Proper BatchNorm, Lower complexity
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class TerminalLSTM(nn.Module):
    """
    SIMPLIFIED TERMINAL LSTM - Defibrillator Protocol
    
    Architecture:
    - BatchNorm on input features
    - Simple unidirectional LSTM
    - Single FC layer with Sigmoid
    
    Designed to prevent gradient issues and sigmoid saturation.
    """
    
    def __init__(
        self, 
        input_size: int = 20,      # Number of features
        hidden_size: int = 128,     # LSTM hidden units
        num_layers: int = 2,        # Stacked LSTM layers
        dropout: float = 0.2        # Regularization
    ):
        super(TerminalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ARCHITECTURE UPGRADE: BatchNorm to stabilize inputs
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # Simple LSTM (not bidirectional)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Simple output head
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            Probability [0, 1] (batch, 1)
        """
        # x shape: (batch, seq_len, features)
        batch, seq, feat = x.shape
        
        # Apply BatchNorm (requires reshaping to 2D)
        x_flat = x.reshape(-1, feat)  # (batch*seq, features)
        x_bn = self.bn_input(x_flat)
        x = x_bn.reshape(batch, seq, feat)  # (batch, seq, features)
        
        # LSTM forward pass
        out, (h_n, c_n) = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        
        # Take last time step output
        last_step = out[:, -1, :]  # (batch, hidden_size)
        
        # Prediction
        prediction = self.fc(last_step)  # (batch, 1)
        return self.sigmoid(prediction)
    
    def predict(self, x: np.ndarray, threshold: float = 0.5) -> Tuple[int, float]:
        """
        Make prediction from numpy array
        
        Args:
            x: Input array (seq_len, features)
            threshold: Classification threshold
            
        Returns:
            signal: 1 for BUY, 0 for SELL/HOLD
            confidence: Probability value
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0)  # Add batch dim
            prob = self.forward(x_tensor).item()
            signal = 1 if prob > threshold else 0
            return signal, prob


class SignalPredictor:
    """
    High-level wrapper for LSTM predictions
    
    Usage:
        predictor = SignalPredictor('models/terminal_lstm.pth')
        result = predictor.predict(features, threshold=0.5)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load(model_path)
    
    def load(self, path: str) -> bool:
        """Load trained model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Reconstruct model with saved config
            self.model = TerminalLSTM(
                input_size=checkpoint.get('input_size', 20),
                hidden_size=checkpoint.get('hidden_size', 128),
                num_layers=checkpoint.get('num_layers', 2),
                dropout=checkpoint.get('dropout', 0.2)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray, threshold: float = 0.55) -> dict:
        """
        Generate trading signal (TOTAL WAR MODE - BUY/SELL/HOLD)
        
        Args:
            features: Array of shape (seq_len, num_features)
            threshold: Base threshold (used for symmetry)
            
        Returns:
            {
                'signal': 'BUY', 'SELL', or 'HOLD',
                'confidence': float,
                'approved': bool
            }
        """
        if self.model is None:
            return {'signal': 'HOLD', 'confidence': 0.0, 'approved': False}
        
        signal_raw, probability = self.model.predict(features, threshold)
        
        # TOTAL WAR SIGNAL LOGIC
        buy_threshold = 0.60   # High probability of increase
        sell_threshold = 0.30  # Low probability = HIGH probability of decrease
        
        if probability >= buy_threshold:
            signal = 'BUY'
            confidence = probability
            approved = True
        elif probability <= sell_threshold:
            signal = 'SELL'
            # Invert confidence for shorts (prob 0.20 -> conf 0.80)
            confidence = 1.0 - probability
            approved = True
        else:
            signal = 'HOLD'
            confidence = probability
            approved = False
        
        return {
            'signal': signal,
            'confidence': confidence,
            'approved': approved
        }


if __name__ == '__main__':
    # Test model creation
    print("🧪 Testing SIMPLIFIED TERMINAL LSTM Model...")
    
    model = TerminalLSTM(input_size=20, hidden_size=128, num_layers=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Architecture:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size, seq_len, features = 32, 60, 20
    x = torch.randn(batch_size, seq_len, features)
    
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n✅ Simplified LSTM Model test passed!")
