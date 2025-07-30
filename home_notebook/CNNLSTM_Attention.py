import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel attention mechanism for CNN features - FIXED for mixed precision"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),  # Enable bias
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True)   # Enable bias
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for CNN features - FIXED for mixed precision"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=True)  # Enable bias
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        return x * attention

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for LSTM sequences - FIXED for mixed precision"""
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Use simple attention for better stability with mixed precision
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=True)
        )
    
    def forward(self, lstm_output):
        # Simple attention mechanism
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum
        attended_output = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return attended_output

class CNNLSTMBranchWithAttention(nn.Module):
    """Enhanced CNN-LSTM branch with attention mechanisms - FIXED for mixed precision"""
    
    def __init__(self, dropout_rate=0.3, use_attention=True):
        super(CNNLSTMBranchWithAttention, self).__init__()
        self.use_attention = use_attention
        
        # CNN layers - ensure all have bias=True for mixed precision stability
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Attention mechanisms for CNN
        if self.use_attention:
            self.channel_attention = ChannelAttention(128)
            self.spatial_attention = SpatialAttention()
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Simplified pooling
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # LSTM layers - use single direction for stability
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=100, batch_first=True, bias=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True, bias=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Attention mechanism for LSTM
        if self.use_attention:
            self.temporal_attention = TemporalAttention(50)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
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
        batch_size = x.size(0)
        
        # CNN feature extraction with attention
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
        
        # Apply CNN attention mechanisms
        if self.use_attention:
            x = self.channel_attention(x)  # Focus on important channels
            x = self.spatial_attention(x)  # Focus on important spatial regions
        
        # Prepare for LSTM
        x = self.adaptive_pool(x)  # (batch, 128, 8, 8)
        x = x.view(batch_size, 128, -1)  # (batch, 128, 64)
        x = x.permute(0, 2, 1)  # (batch, 64, 128)
        
        # LSTM processing
        x, _ = self.lstm1(x)  # (batch, 64, 100)
        x = self.lstm_dropout(x)
        
        x, _ = self.lstm2(x)  # (batch, 64, 50)
        x = self.lstm_dropout(x)
        
        # Apply temporal attention
        if self.use_attention:
            features = self.temporal_attention(x)  # (batch, 50)
        else:
            # Fallback: use last timestep
            features = x[:, -1, :]  # (batch, 50)
        
        return features

class CNNLSTMIQModelWithAttention(nn.Module):
    """CNN-LSTM model with attention mechanisms - FIXED for mixed precision training"""
    
    def __init__(self, num_classes, dropout_rate=0.4, use_attention=True):
        super(CNNLSTMIQModelWithAttention, self).__init__()
        
        # Separate branches for I and Q signals
        self.i_branch = CNNLSTMBranchWithAttention(
            dropout_rate=dropout_rate * 0.75, 
            use_attention=use_attention
        )
        self.q_branch = CNNLSTMBranchWithAttention(
            dropout_rate=dropout_rate * 0.75, 
            use_attention=use_attention
        )
        
        # Feature fusion with attention
        feature_dim = 50  # From LSTM output
        if use_attention:
            self.fusion_attention = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim, bias=True),
                nn.Tanh(),
                nn.Linear(feature_dim, 2, bias=True),
                nn.Softmax(dim=1)
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            
            nn.Linear(128, 64, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, num_classes, bias=True)
        )
        
        self.use_attention = use_attention
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
        Forward pass with attention-enhanced I/Q processing
        
        Args:
            i_input: I signal tensor of shape (batch, 1, H, W)
            q_input: Q signal tensor of shape (batch, 1, H, W)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Process I and Q signals separately
        i_features = self.i_branch(i_input)  # (batch, 50)
        q_features = self.q_branch(q_input)  # (batch, 50)
        
        # Feature fusion with attention
        if self.use_attention:
            # Concatenate I and Q features
            concat_features = torch.cat([i_features, q_features], dim=1)  # (batch, 100)
            
            # Learn attention weights for I and Q
            fusion_weights = self.fusion_attention(concat_features)  # (batch, 2)
            
            # Apply attention weights
            attended_i = i_features * fusion_weights[:, 0:1]
            attended_q = q_features * fusion_weights[:, 1:2]
            
            combined_features = attended_i + attended_q  # (batch, 50)
        else:
            # Simple addition fusion
            combined_features = i_features + q_features  # (batch, 50)
        
        # Classification
        output = self.classifier(combined_features)
        return output

def create_attention_CNNLSTMIQModel(n_labels=9, dropout_rate=0.5, use_attention=True):  
    """
    Create CNN-LSTM model with attention mechanisms - FIXED for mixed precision
    
    Args:
        n_labels: Number of modulation classes
        dropout_rate: Dropout rate 
        use_attention: Whether to use attention mechanisms
    """
    return CNNLSTMIQModelWithAttention(
        num_classes=n_labels, 
        dropout_rate=dropout_rate, 
        use_attention=use_attention
    )

# debugging mixed precision issues
def check_model_precision(model):
    """Debug function to check model parameter types"""
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

# Training-compatible model creation
def create_model_for_training(n_labels=9, dropout_rate=0.4, device='cuda'):
    """Create model properly configured for mixed precision training"""
    model = create_attention_CNNLSTMIQModel(
        n_labels=n_labels, 
        dropout_rate=dropout_rate, 
        use_attention=True
    )
    
    # Move to device and ensure proper dtype
    model = model.to(device)
    
    # Ensure all parameters are in float32 initially
    model = model.float()
    
    return model