import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    """CNN branch for spatial feature extraction"""
    
    def __init__(self, dropout_rate=0.3):
        super(CNNBranch, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Additional conv layer for more features
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Global pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FC layer to get desired feature dimension
        self.fc = nn.Linear(256, 128)
        
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
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Global pooling and FC
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        x = self.fc(x)  # (batch, 128)
        
        return x


class LSTMBranch(nn.Module):
    """LSTM branch for sequential feature extraction"""
    
    def __init__(self, dropout_rate=0.3):
        super(LSTMBranch, self).__init__()
        
        # Input projection layer
        self.input_projection = nn.Linear(32, 64)  # Assuming input width of 32
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=100, batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=64, batch_first=True, dropout=dropout_rate)
        
        # Additional processing
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention_layer = nn.Linear(64, 1)
        
        # Final projection to match CNN output dimension
        self.fc = nn.Linear(64, 128)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
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
        batch_size = x.size(0)
        
        # Convert 2D input to sequence: treat each row as a time step
        # x shape: (batch, 1, H, W) -> (batch, H, W)
        x = x.squeeze(1)  # Remove channel dimension
        
        # Project each row to higher dimension
        x = self.input_projection(x)  # (batch, H, 64)
        
        # LSTM processing
        x, _ = self.lstm1(x)  # (batch, H, 100)
        x = self.lstm_dropout(x)
        
        x, _ = self.lstm2(x)  # (batch, H, 64)
        x = self.lstm_dropout(x)
        
        # Attention mechanism
        attention_scores = self.attention_layer(x)  # (batch, H, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, H, 1)
        features = torch.sum(x * attention_weights, dim=1)  # (batch, 64)
        
        # Final projection
        features = self.fc(features)  # (batch, 128)
        
        return features


class ParallelCNNLSTMBranch(nn.Module):
    """Parallel CNN-LSTM branch combining both approaches"""
    
    def __init__(self, dropout_rate=0.3, fusion_method='concat'):
        super(ParallelCNNLSTMBranch, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Parallel branches
        self.cnn_branch = CNNBranch(dropout_rate)
        self.lstm_branch = LSTMBranch(dropout_rate)
        
        # Fusion layer
        if fusion_method == 'concat':
            self.fusion_fc = nn.Sequential(
                nn.Linear(256, 128),  # 128 + 128 = 256
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64)
            )
        elif fusion_method == 'add':
            # No additional layer needed, features are same size (128)
            self.fusion_fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout_rate)
            )
        elif fusion_method == 'multiply':
            # Element-wise multiplication
            self.fusion_fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout_rate)
            )
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.attention_cnn = nn.Linear(128, 1)
            self.attention_lstm = nn.Linear(128, 1)
            self.fusion_fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout_rate)
            )
        
        self._initialize_fusion_weights()
    
    def _initialize_fusion_weights(self):
        for m in self.fusion_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Parallel processing
        cnn_features = self.cnn_branch(x)      # (batch, 128)
        lstm_features = self.lstm_branch(x)    # (batch, 128)
        
        # Fusion
        if self.fusion_method == 'concat':
            combined = torch.cat([cnn_features, lstm_features], dim=1)  # (batch, 256)
        elif self.fusion_method == 'add':
            combined = cnn_features + lstm_features  # (batch, 128)
        elif self.fusion_method == 'multiply':
            combined = cnn_features * lstm_features  # (batch, 128)
        elif self.fusion_method == 'attention':
            # Attention-based weighting
            cnn_weight = torch.sigmoid(self.attention_cnn(cnn_features))  # (batch, 1)
            lstm_weight = torch.sigmoid(self.attention_lstm(lstm_features))  # (batch, 1)
            
            # Normalize weights
            total_weight = cnn_weight + lstm_weight
            cnn_weight = cnn_weight / total_weight
            lstm_weight = lstm_weight / total_weight
            
            combined = cnn_weight * cnn_features + lstm_weight * lstm_features  # (batch, 128)
        
        # Final processing
        output = self.fusion_fc(combined)
        
        return output


class ParallelCNNLSTMIQModel(nn.Module):
    """Parallel CNN-LSTM model for I/Q signal processing"""
    
    def __init__(self, num_classes, dropout_rate=0.4, fusion_method='concat'):
        super(ParallelCNNLSTMIQModel, self).__init__()
        
        # Separate parallel branches for I and Q signals
        self.i_branch = ParallelCNNLSTMBranch(dropout_rate=dropout_rate * 0.75, fusion_method=fusion_method)
        self.q_branch = ParallelCNNLSTMBranch(dropout_rate=dropout_rate * 0.75, fusion_method=fusion_method)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),  # Output from each branch is 64
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
        
        self._initialize_classifier_weights()
    
    def _initialize_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, i_input, q_input):
        """
        Forward pass with parallel CNN-LSTM processing
        
        Args:
            i_input: I signal tensor of shape (batch, 1, 32, 32)
            q_input: Q signal tensor of shape (batch, 1, 32, 32)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Process I and Q signals through parallel CNN-LSTM branches
        i_features = self.i_branch(i_input)  # Shape: (batch, 64)
        q_features = self.q_branch(q_input)  # Shape: (batch, 64)
        
        # Combine I and Q features (using addition like original)
        combined_features = i_features + q_features  # Shape: (batch, 64)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output


def create_parallel_CNNLSTMIQModel(n_labels=9, dropout_rate=0.5, fusion_method='concat'):
    """
    Create parallel CNN-LSTM model
    
    Args:
        n_labels: Number of output classes
        dropout_rate: Dropout rate
        fusion_method: How to fuse CNN and LSTM features
                      Options: 'concat', 'add', 'multiply', 'attention'
    
    Returns:
        ParallelCNNLSTMIQModel instance
    """
    return ParallelCNNLSTMIQModel(n_labels, dropout_rate, fusion_method)

