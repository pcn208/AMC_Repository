import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMBranch(nn.Module):
    """Enhanced CNN-LSTM branch for processing I or Q signals separately"""
    
    def __init__(self, dropout_rate=0.3):
        super(CNNLSTMBranch, self).__init__()
        
        # CNN layers (keeping your existing structure)
        # First conv block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # Second conv block  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # LSTM layers (NEW - inspired by the paper)
        # After CNN, we'll have spatial-temporal features to feed into LSTM
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Calculate the sequence length after convolutions
        # This depends on your input size - adjust accordingly
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Fixed spatial size for LSTM
        
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
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction (your existing structure)
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
        
        # Prepare for LSTM: reshape CNN output to sequence
        # x shape: (batch, 128, H, W)
        x = self.adaptive_pool(x)  # (batch, 128, 8, 8)
        
        # Flatten spatial dimensions to create sequence
        # Option 1: Treat each spatial location as a time step
        x = x.view(batch_size, 128, -1)  # (batch, 128, 64)
        x = x.permute(0, 2, 1)  # (batch, 64, 128) - (batch, seq_len, features)
        
        # LSTM processing (NEW - following paper's approach)
        x, _ = self.lstm1(x)  # (batch, 64, 100)
        x = self.lstm_dropout(x)
        
        x, _ = self.lstm2(x)  # (batch, 64, 50)
        x = self.lstm_dropout(x)
        
        # Take the last time step output (following paper's approach)
        features = x[:, -1, :]  # (batch, 50)
        
        return features


class CNNLSTMIQModel(nn.Module):
    """Enhanced CNN-LSTM model for I/Q signal processing with your addition fusion"""
    
    def __init__(self, num_classes, dropout_rate=0.4):
        super(CNNLSTMIQModel, self).__init__()
        
        # Separate branches for I and Q signals (enhanced with LSTM)
        self.i_branch = CNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        self.q_branch = CNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        
        # Classifier after combining features
        # Since we're adding I and Q features, output size is still 50 (from LSTM)
        self.classifier = nn.Sequential(
            nn.Linear(50, 256),  # LSTM output dimension
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
        Forward pass with separate I and Q processing enhanced with LSTM
        
        Args:
            i_input: I signal tensor of shape (batch, 1, 32, 32)
            q_input: Q signal tensor of shape (batch, 1, 32, 32)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Process I and Q signals separately through CNN-LSTM
        i_features = self.i_branch(i_input)  # Shape: (batch, 50)
        q_features = self.q_branch(q_input)  # Shape: (batch, 50)
        
        # Combine features using addition (keeping your approach)
        combined_features = i_features + q_features  # Shape: (batch, 50)
        
        # Apply classifier
        output = self.classifier(combined_features)
        
        return output


# Alternative: Outer Product Fusion (more like the paper)
class CNNLSTMIQModelOuterProduct(nn.Module):
    """CNN-LSTM model with outer product fusion like the paper"""
    
    def __init__(self, num_classes, dropout_rate=0.4):
        super(CNNLSTMIQModelOuterProduct, self).__init__()
        
        self.i_branch = CNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        self.q_branch = CNNLSTMBranch(dropout_rate=dropout_rate * 0.75)
        
        # Outer product creates 50x50 = 2500 features
        self.classifier = nn.Sequential(
            nn.Linear(2500, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            
            nn.Linear(256, num_classes)
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
        # Extract features
        i_features = self.i_branch(i_input)  # (batch, 50)
        q_features = self.q_branch(q_input)  # (batch, 50)
        
        # Outer product fusion (like the paper)
        outer_product = torch.bmm(
            i_features.unsqueeze(2),  # (batch, 50, 1)
            q_features.unsqueeze(1)   # (batch, 1, 50)
        )  # Result: (batch, 50, 50)
        
        # Flatten for classification
        combined_features = outer_product.view(outer_product.size(0), -1)  # (batch, 2500)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output


def create_enhanced_CNNLSTMIQModel(n_labels=8, dropout_rate=0.5, use_outer_product=False):  
    """Create enhanced CNN-LSTM model"""
    if use_outer_product:
        return CNNLSTMIQModelOuterProduct(n_labels, dropout_rate)
    else:
        return CNNLSTMIQModel(n_labels, dropout_rate)