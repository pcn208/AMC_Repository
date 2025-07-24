import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_1(nn.Module):
    """Shallow CNN branch - improved with better regularization"""
    def __init__(self, dropout_rate=0.3):
        super(CNN_1, self).__init__()
        
        # Reduced channel sizes to prevent overfitting
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Added adaptive pooling for consistent output
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))  # More spatial info than (1,1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # Output: (batch, 256)

class CNN_2(nn.Module):
    """Deeper CNN branch - improved with better regularization"""
    def __init__(self, dropout_rate=0.3):
        super(CNN_2, self).__init__()
        
        # Reduced channel sizes
        self.conv3 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # More spatial info retention
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # Output: (batch, 512)


class ImprovedParallelBranch(nn.Module):
    """Single branch that processes one channel (I or Q) with anti-overfitting measures"""
    def __init__(self, dropout_rate=0.4):
        super(ImprovedParallelBranch, self).__init__()
        
        # Parallel CNN processing
        self.cnn1 = CNN_1(dropout_rate)
        self.cnn2 = CNN_2(dropout_rate)
        
        # Feature fusion with reduced dimensions
        fused_feature_size = 256 + 512  # 768
        
        # Add compression layer to reduce overfitting
        self.compression = nn.Sequential(
            nn.Linear(fused_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Simplified LSTM with proper sequence processing
        self.feature_reshape = nn.Linear(256, 64)  # Create proper features for LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Attention mechanism instead of just last timestep
        self.attention = nn.Linear(128, 1)  # 64*2 from bidirectional

    def forward(self, x):
        # Parallel CNN feature extraction
        features1 = self.cnn1(x)  # (batch, 256)
        features2 = self.cnn2(x)  # (batch, 512)
        
        # Fuse features
        fused_features = torch.cat((features1, features2), dim=1)  # (batch, 768)
        
        # Compress to prevent overfitting
        compressed = self.compression(fused_features)  # (batch, 256)
        
        # Create sequence for LSTM (simulate temporal structure)
        batch_size = compressed.size(0)
        # Split features into temporal chunks
        seq_features = self.feature_reshape(compressed)  # (batch, 64)
        
        # Create artificial sequence by reshaping
        # This simulates temporal dependencies in the feature space
        # NOTE: This part is complex and might be a source of issues if not handled carefully.
        # The logic here seems to create a fixed sequence length of 4.
        seq_input = seq_features.view(batch_size, 4, 16)  # (batch, seq_len=4, features=16)
        
        # Pad to minimum LSTM input size
        # This padding seems arbitrary and might not be optimal.
        seq_input = F.pad(seq_input, (0, 48))  # Pad to 64 features: (batch, 4, 64)
        
        # LSTM processing
        lstm_out, _ = self.lstm(seq_input)  # (batch, 4, 128)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention-based pooling instead of last timestep
        attention_scores = self.attention(lstm_out)  # (batch, 4, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, 4, 1)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, 128)
        
        return attended_features


class FixedDiagramIQModel(nn.Module):
    """Fixed I/Q model with shared weights and better fusion to prevent overfitting"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(FixedDiagramIQModel, self).__init__()
        
        # SHARED branch to reduce parameters and prevent overfitting
        self.shared_branch = ImprovedParallelBranch(dropout_rate * 0.75)
        
        # Cross-channel interaction layer
        self.cross_interaction = nn.Sequential(
            nn.Linear(256, 128),  # 128*2 from I and Q branches
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Final classifier with heavy regularization
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            
            nn.Linear(32, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, i_input, q_input):
        # Process I and Q through SHARED branch (weight sharing reduces overfitting)
        i_features = self.shared_branch(i_input)    # (batch, 128)
        q_features = self.shared_branch(q_input)    # (batch, 128)
        
        # Concatenate for richer interaction
        combined_features = torch.cat([i_features, q_features], dim=1)  # (batch, 256)
        
        # Cross-channel interaction
        interacted_features = self.cross_interaction(combined_features)  # (batch, 128)
        
        # Final classification
        output = self.classifier(interacted_features)
        
        return output


class AlternativeParallelModel(nn.Module):
    """Alternative parallel model with different anti-overfitting strategy"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(AlternativeParallelModel, self).__init__()
        
        # Separate but lighter branches
        self.i_branch = ImprovedParallelBranch(dropout_rate * 0.8)
        self.q_branch = ImprovedParallelBranch(dropout_rate * 0.8)
        
        # Feature selection layer (learnable attention between I and Q)
        self.feature_selector = nn.Sequential(
            nn.Linear(256, 64),  # 128*2
            nn.Tanh(),
            nn.Linear(64, 2),    # Weights for I and Q features
            nn.Softmax(dim=1)
        )
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, i_input, q_input):
        # Extract features
        i_features = self.i_branch(i_input)  # (batch, 128)
        q_features = self.q_branch(q_input)  # (batch, 128)
        
        # Stack for selection
        stacked_features = torch.stack([i_features, q_features], dim=2)  # (batch, 128, 2)
        
        # Learn feature selection weights
        concat_for_selection = torch.cat([i_features, q_features], dim=1)  # (batch, 256)
        selection_weights = self.feature_selector(concat_for_selection)  # (batch, 2)
        selection_weights = selection_weights.unsqueeze(1)  # (batch, 1, 2)
        
        # Apply learned weighting
        weighted_features = torch.sum(stacked_features * selection_weights, dim=2)  # (batch, 128)
        
        # Classify
        output = self.classifier(weighted_features)
        return output


def create_fixed_parallel_model(num_classes=2, dropout_rate=0.5, use_alternative=False):
    """
    Create fixed parallel model with anti-overfitting measures
    
    Args:
        num_classes: Number of output classes (e.g., 2 for 8PSK vs 16QAM)
        dropout_rate: Dropout rate for regularization
        use_alternative: Use alternative model architecture
    
    Returns:
        A PyTorch model instance.
    """
    if use_alternative:
        return AlternativeParallelModel(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        return FixedDiagramIQModel(num_classes=num_classes, dropout_rate=dropout_rate)


# For backward compatibility
def create_diagram_iq_model(num_classes=2, dropout_rate=0.5):
    """
    Backward compatible function - now returns the fixed model.
    This function now uses 'num_classes' to be consistent with the rest of the code.
    """
    return create_fixed_parallel_model(num_classes=num_classes, dropout_rate=dropout_rate, use_alternative=False)
