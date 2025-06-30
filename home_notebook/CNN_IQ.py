import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNIQBranch(nn.Module):
    """Individual CNN branch for processing I or Q signals separately"""
    
    def __init__(self, dropout_rate=0.3):
        super(CNNIQBranch, self).__init__()
        
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
        
        # Global average pooling to reduce parameters
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
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
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.flatten(1)  # Shape: (batch, 128)
        
        return x


class CNNIQModel(nn.Module):
    """Separate branch CNN model for I/Q signal processing with addition fusion"""
    
    def __init__(self, num_classes, dropout_rate=0.4):
        super(CNNIQModel, self).__init__()
        
        # Separate branches for I and Q signals
        self.i_branch = CNNIQBranch(dropout_rate=dropout_rate * 0.75)
        self.q_branch = CNNIQBranch(dropout_rate=dropout_rate * 0.75)
        
        # Classifier after combining features
        # Since we're adding I and Q features, output size is still 128
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),  # Combined features dimension
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
        Forward pass with separate I and Q processing
        
        Args:
            i_input: I signal tensor of shape (batch, 1, 32, 32)
            q_input: Q signal tensor of shape (batch, 1, 32, 32)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Process I and Q signals separately
        i_features = self.i_branch(i_input)  # Shape: (batch, 128)
        q_features = self.q_branch(q_input)  # Shape: (batch, 128)
        
        # Combine features using addition
        combined_features = i_features + q_features  # Shape: (batch, 128)
        
        # Apply classifier
        output = self.classifier(combined_features)
        
        return output
def create_CNNIQModel(n_labels=8, dropout_rate=0.5):  
    return CNNIQModel(n_labels, dropout_rate)