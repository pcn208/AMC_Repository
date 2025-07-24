import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletCNNLSTMBranch(nn.Module):
    """Enhanced CNN-LSTM branch optimized for wavelet coefficient processing"""
    
    def __init__(self, dropout_rate=0.3):
        super(WaveletCNNLSTMBranch, self).__init__()
        
        # Enhanced CNN layers for wavelet features
        # Wavelet coefficients have time-frequency structure, so we need to preserve that
        
        # First conv block - smaller kernels for fine-grained wavelet features
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # Second conv block - capture multi-scale wavelet patterns
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Third conv block - deeper feature extraction for complex wavelet patterns
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Gentler pooling for wavelets
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Adaptive pooling to create consistent sequence for LSTM
        # After 3 pooling ops: 32->16->8->4, so we'll use 4x4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # LSTM layers optimized for wavelet time-frequency sequences
        # We'll treat spatial locations as time steps
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)  # 128*2=256
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Attention mechanism for wavelet feature selection
        self.attention_layer = nn.Linear(128, 1)  # 64*2=128 from bidirectional LSTM
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for wavelet coefficients
        x: (batch, 1, 32, 32) - wavelet coefficients
        """
        batch_size = x.size(0)
        
        # CNN feature extraction for wavelet patterns
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        x = self.dropout(x)
        
        # Third conv block for complex wavelet patterns
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        x = self.dropout(x)
        
        # Ensure consistent spatial size
        x = self.adaptive_pool(x)  # (batch, 256, 4, 4)
        
        # Prepare for LSTM: treat spatial locations as sequence
        # Reshape to sequence: (batch, seq_len, features)
        x = x.view(batch_size, 256, -1)  # (batch, 256, 16)
        x = x.permute(0, 2, 1)  # (batch, 16, 256)
        
        # Bidirectional LSTM for capturing wavelet temporal dependencies
        x, _ = self.lstm1(x)  # (batch, 16, 256) - bidirectional doubles output
        x = self.lstm_dropout(x)
        
        x, _ = self.lstm2(x)  # (batch, 16, 128) - bidirectional
        x = self.lstm_dropout(x)
        
        # Global attention mechanism for wavelet feature selection
        attention_scores = self.attention_layer(x)  # (batch, 16, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, 16, 1)
        features = torch.sum(x * attention_weights, dim=1)  # (batch, 128)
        
        return features


class WaveletCNNLSTMIQModel(nn.Module):
    """CNN-LSTM model optimized for IQ wavelet coefficient processing"""
    
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(WaveletCNNLSTMIQModel, self).__init__()
        
        # Separate branches for I and Q wavelet coefficients
        self.i_branch = WaveletCNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        self.q_branch = WaveletCNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(256, 128),  # 128 + 128 from each branch
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Enhanced classifier for wavelet features
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.fusion_layer, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, i_wavelet, q_wavelet):
        """
        Forward pass for IQ wavelet coefficients
        
        Args:
            i_wavelet: I channel wavelet coefficients (batch, 1, 32, 32)
            q_wavelet: Q channel wavelet coefficients (batch, 1, 32, 32)
        
        Returns:
            Output logits (batch, num_classes)
        """
        # Process I and Q wavelet coefficients separately
        i_features = self.i_branch(i_wavelet)  # (batch, 128)
        q_features = self.q_branch(q_wavelet)  # (batch, 128)
        
        # Concatenate features for richer representation
        combined_features = torch.cat([i_features, q_features], dim=1)  # (batch, 256)
        
        # Fusion layer
        fused_features = self.fusion_layer(combined_features)  # (batch, 128)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output


class WaveletCNNLSTMIQModelWithAttention(nn.Module):
    """Advanced version with cross-attention between I and Q branches"""
    
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(WaveletCNNLSTMIQModelWithAttention, self).__init__()
        
        self.i_branch = WaveletCNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        self.q_branch = WaveletCNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        
        # Cross-attention between I and Q features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=8, 
            dropout=dropout_rate * 0.5,
            batch_first=True
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),  # 128*2 after attention
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, i_wavelet, q_wavelet):
        # Extract features from both branches
        i_features = self.i_branch(i_wavelet)  # (batch, 128)
        q_features = self.q_branch(q_wavelet)  # (batch, 128)
        
        # Prepare for cross-attention (need sequence dimension)
        i_seq = i_features.unsqueeze(1)  # (batch, 1, 128)
        q_seq = q_features.unsqueeze(1)  # (batch, 1, 128)
        
        # Cross-attention: I attends to Q and vice versa
        i_attended, _ = self.cross_attention(i_seq, q_seq, q_seq)  # (batch, 1, 128)
        q_attended, _ = self.cross_attention(q_seq, i_seq, i_seq)  # (batch, 1, 128)
        
        # Combine attended features
        combined = torch.cat([
            i_attended.squeeze(1), 
            q_attended.squeeze(1)
        ], dim=1)  # (batch, 256)
        
        # Classification
        output = self.classifier(combined)
        
        return output


def create_wavelet_model(num_classes=2, dropout_rate=0.4, use_attention=False):
    """
    Create CNN-LSTM model optimized for IQ wavelet features
    
    Args:
        num_classes: Number of output classes (2 for 8PSK vs 16QAM)
        dropout_rate: Dropout rate for regularization
        use_attention: Whether to use cross-attention between I/Q branches
    
    Returns:
        PyTorch model optimized for wavelet coefficient processing
    """
    if use_attention:
        return WaveletCNNLSTMIQModelWithAttention(num_classes, dropout_rate)
    else:
        return WaveletCNNLSTMIQModel(num_classes, dropout_rate)
