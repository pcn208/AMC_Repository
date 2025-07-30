import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    """CNN Branch with dual shallow and deep paths"""
    def __init__(self, dropout_rate=0.3):
        super(CNNBranch, self).__init__()
        
        # CNN_1 (Shallow path)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # CNN_2 (Deep path)
        self.conv3 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Common layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # CNN_1 (Shallow path)
        x1 = self.leaky_relu(self.bn1(self.conv1(x)))
        x1 = self.leaky_relu(self.bn2(self.conv2(x1)))
        x1 = self.pool(x1)
        x1 = self.dropout(x1)
        x1 = self.global_pool(x1)
        features1 = x1.view(x1.size(0), -1)  # 128 features
        
        # CNN_2 (Deep path)
        x2 = self.leaky_relu(self.bn3(self.conv3(x)))
        x2 = self.leaky_relu(self.bn4(self.conv4(x2)))
        x2 = self.pool(x2)
        x2 = self.dropout(x2)
        
        x2 = self.leaky_relu(self.bn5(self.conv5(x2)))
        x2 = self.leaky_relu(self.bn6(self.conv6(x2)))
        x2 = self.pool(x2)
        x2 = self.dropout(x2)
        x2 = self.global_pool(x2)
        features2 = x2.view(x2.size(0), -1)  # 256 features
        
        # Concatenate features
        fused_features = torch.cat((features1, features2), dim=1)  # 384 features
        return fused_features

class LSTMBranch(nn.Module):
    """LSTM Branch for temporal processing"""
    def __init__(self, input_size, dropout_rate=0.3):
        super(LSTMBranch, self).__init__()
        self.input_size = input_size
        
        # For sequence preparation from 2D input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Create 8x8 = 64 sequence length
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Prepare sequence from 2D input
        # x shape: (batch, 1, H, W)
        x = self.adaptive_pool(x)  # (batch, 1, 8, 8)
        
        # Reshape for LSTM: treat spatial dimensions as sequence
        batch_size = x.size(0)
        x = x.view(batch_size, 8, 8)  # (batch, seq_len=8, features=8)
        
        # LSTM processing
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.lstm_dropout(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Use last time step
        last_time_step = lstm_out[:, -1, :]  # (batch, 128)
        
        return last_time_step

class ParallelCNNLSTMModel(nn.Module):
    """Parallel CNN-LSTM Model for I/Q Signal Processing"""
    def __init__(self, num_classes, dropout_rate=0.4):
        super(ParallelCNNLSTMModel, self).__init__()
        
        # I-Signal branches
        self.i_cnn_branch = CNNBranch(dropout_rate)
        self.i_lstm_branch = LSTMBranch(input_size=64, dropout_rate=dropout_rate)
        
        # Q-Signal branches  
        self.q_cnn_branch = CNNBranch(dropout_rate)
        self.q_lstm_branch = LSTMBranch(input_size=64, dropout_rate=dropout_rate)
        
        # Final fusion and classification
        # Total features: I-CNN(384) + I-LSTM(128) + Q-CNN(384) + Q-LSTM(128) = 1024
        self.final_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, i_input, q_input):
        # Process I-Signal in parallel
        i_cnn_features = self.i_cnn_branch(i_input)      # 384 features
        i_lstm_features = self.i_lstm_branch(i_input)    # 128 features
        
        # Process Q-Signal in parallel  
        q_cnn_features = self.q_cnn_branch(q_input)      # 384 features
        q_lstm_features = self.q_lstm_branch(q_input)    # 128 features
        
        # Global feature fusion
        global_features = torch.cat([
            i_cnn_features,   # 384
            i_lstm_features,  # 128
            q_cnn_features,   # 384
            q_lstm_features   # 128
        ], dim=1)  # Total: 1024 features
        
        # Final classification
        output = self.final_classifier(global_features)
        return output

class EnhancedParallelCNNLSTMModel(nn.Module):
    """Enhanced version with attention mechanism for better feature fusion"""
    def __init__(self, num_classes, dropout_rate=0.4):
        super(EnhancedParallelCNNLSTMModel, self).__init__()
        
        # Same parallel branches
        self.i_cnn_branch = CNNBranch(dropout_rate)
        self.i_lstm_branch = LSTMBranch(input_size=64, dropout_rate=dropout_rate)
        self.q_cnn_branch = CNNBranch(dropout_rate)
        self.q_lstm_branch = LSTMBranch(input_size=64, dropout_rate=dropout_rate)
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature projection layers
        self.i_cnn_proj = nn.Linear(384, 256)
        self.i_lstm_proj = nn.Linear(128, 256)
        self.q_cnn_proj = nn.Linear(384, 256)
        self.q_lstm_proj = nn.Linear(128, 256)
        
        # Final classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, i_input, q_input):
        # Process all branches in parallel
        i_cnn_features = self.i_cnn_branch(i_input)
        i_lstm_features = self.i_lstm_branch(i_input)
        q_cnn_features = self.q_cnn_branch(q_input)
        q_lstm_features = self.q_lstm_branch(q_input)
        
        # Project to common dimension
        i_cnn_proj = self.i_cnn_proj(i_cnn_features).unsqueeze(1)    # (batch, 1, 256)
        i_lstm_proj = self.i_lstm_proj(i_lstm_features).unsqueeze(1)  # (batch, 1, 256)
        q_cnn_proj = self.q_cnn_proj(q_cnn_features).unsqueeze(1)    # (batch, 1, 256)
        q_lstm_proj = self.q_lstm_proj(q_lstm_features).unsqueeze(1)  # (batch, 1, 256)
        
        # Stack features for attention
        features = torch.cat([i_cnn_proj, i_lstm_proj, q_cnn_proj, q_lstm_proj], dim=1)  # (batch, 4, 256)
        
        # Apply attention
        attended_features, _ = self.attention(features, features, features)  # (batch, 4, 256)
        
        # Global average pooling across the 4 feature types
        fused_features = attended_features.mean(dim=1)  # (batch, 256)
        
        # Final classification
        output = self.final_classifier(fused_features)
        return output

def create_parallel_cnn_lstm_model(n_labels=9, dropout_rate=0.4, enhanced=False):
    """
    Creates the Parallel CNN-LSTM model instance.
    
    Args:
        n_labels: Number of output classes.
        dropout_rate: The base dropout rate for the model.
        enhanced: Whether to use the enhanced version with attention.
    
    Returns:
        An instance of ParallelCNNLSTMModel or EnhancedParallelCNNLSTMModel.
    """
    if enhanced:
        return EnhancedParallelCNNLSTMModel(num_classes=n_labels, dropout_rate=dropout_rate)
    else:
        return ParallelCNNLSTMModel(num_classes=n_labels, dropout_rate=dropout_rate)