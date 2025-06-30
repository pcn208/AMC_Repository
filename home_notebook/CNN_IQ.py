import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNIQBranch(nn.Module):
    """Individual CNN branch for processing I or Q signals separately with residual connections and attention"""
    
    def __init__(self, dropout_rate=0.3):
        super(CNNIQBranch, self).__init__()
        
        # First conv block with residual connection
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # Residual connection for first block (1->64 channels)
        self.residual1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.bn_residual1 = nn.BatchNorm2d(64)
        
        # Channel attention for first block
        self.channel_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64//4, 64, 1),
            nn.Sigmoid()
        )
        
        # Second conv block with residual connection
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Residual connection for second block (64->128 channels)
        self.residual2 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.bn_residual2 = nn.BatchNorm2d(128)
        
        # Channel attention for second block
        self.channel_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128//4, 128, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention block for fine-grained phase features
        self.spatial_att = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
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
        # Store input for residual connection
        identity1 = x
        
        # First conv block with residual connection
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        
        # Add residual connection (project identity to match channels)
        identity1_proj = self.residual1(identity1)
        identity1_proj = self.bn_residual1(identity1_proj)
        x = x + identity1_proj
        x = self.leaky_relu(x)
        
        # Apply channel attention to first block
        att_weights1 = self.channel_att1(x)
        x = x * att_weights1
        
        x = self.pool(x)
        x = self.dropout(x)
        
        # Store for second residual connection
        identity2 = x
        
        # Second conv block with residual connection
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        
        # Add residual connection (project identity to match channels)
        identity2_proj = self.residual2(identity2)
        identity2_proj = self.bn_residual2(identity2_proj)
        x = x + identity2_proj
        x = self.leaky_relu(x)
        
        # Apply channel attention to second block
        att_weights2 = self.channel_att2(x)
        x = x * att_weights2
        
        # Apply spatial attention for fine-grained features
        spatial_weights = self.spatial_att(x)
        x = x * spatial_weights
        
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
        
        # Enhanced classifier with residual connection
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
        
        # Classifier residual connection (128 -> 64 skip)
        self.classifier_residual = nn.Linear(128, 64)
        
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
        
        # Initialize residual connection
        nn.init.kaiming_normal_(self.classifier_residual.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.classifier_residual.bias is not None:
            nn.init.constant_(self.classifier_residual.bias, 0)
    
    def forward(self, i_input, q_input):
        """
        Forward pass with separate I and Q processing
        
        Args:
            i_input: I signal tensor of shape (batch, 1, H, W)
            q_input: Q signal tensor of shape (batch, 1, H, W)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Process I and Q signals separately
        i_features = self.i_branch(i_input)  # Shape: (batch, 128)
        q_features = self.q_branch(q_input)  # Shape: (batch, 128)
        
        # Combine features using addition
        combined_features = i_features + q_features  # Shape: (batch, 128)
        
        # Enhanced classifier with residual connection
        x = combined_features
        
        # First part of classifier (128 -> 256 -> 128)
        x = self.classifier[0](x)  # Linear(128, 256)
        x = self.classifier[1](x)  # BatchNorm1d
        x = self.classifier[2](x)  # LeakyReLU
        x = self.classifier[3](x)  # Dropout
        
        x = self.classifier[4](x)  # Linear(256, 128)
        x = self.classifier[5](x)  # BatchNorm1d
        x = self.classifier[6](x)  # LeakyReLU
        x = self.classifier[7](x)  # Dropout
        
        # Second part (128 -> 64) with residual connection
        identity = self.classifier_residual(combined_features)  # Skip connection
        
        x = self.classifier[8](x)  # Linear(128, 64)
        x = x + identity  # Add residual connection
        x = self.classifier[9](x)  # LeakyReLU
        x = self.classifier[10](x)  # Dropout
        
        # Final classification layer
        output = self.classifier[11](x)  # Linear(64, num_classes)
        
        return output

def create_CNNIQModel(n_labels=6, dropout_rate=0.5):  
    return CNNIQModel(n_labels, dropout_rate)