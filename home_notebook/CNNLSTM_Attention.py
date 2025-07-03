import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel attention mechanism for CNN features"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
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
    """Spatial attention mechanism for CNN features"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
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
    """Temporal attention mechanism for LSTM sequences - FIXED & IMPROVED"""
    def __init__(self, hidden_size, num_heads=None):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.use_multihead = True
        
        # Robust auto-head selection
        if num_heads is None:
            # Find ALL possible divisors within reasonable limits
            max_heads = min(16, hidden_size)  # Don't exceed 16 or hidden_size
            valid_heads = [h for h in range(1, max_heads+1) if hidden_size % h == 0]
            
            if valid_heads:
                # Prefer heads close to sqrt(hidden_size) for balance
                ideal = int(round(hidden_size**0.5))
                valid_heads.sort(key=lambda x: abs(x-ideal))
                num_heads = valid_heads[0]  # Closest to ideal
            else:
                # Fallback to simple attention if no valid heads
                num_heads = 1
                self.use_multihead = False
                print(f"⚠️ TemporalAttention: No valid heads for {hidden_size}-dim features. Using simple attention.")

        # Final compatibility check
        if hidden_size % num_heads != 0:
            self.use_multihead = False
            print(f"⚠️ TemporalAttention: {hidden_size} not divisible by {num_heads}. Using simple attention.")
        else:
            self.num_heads = num_heads
            print(f"✅ TemporalAttention: Using {num_heads} heads for {hidden_size}-dim features")
        
        # Multi-head attention only if compatible
        if self.use_multihead:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=hidden_size, 
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
        
        # Simple attention (always available as fallback)
        self.simple_attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, lstm_output):
        if hasattr(self, 'multihead_attn') and self.use_multihead:
            # Multi-head attention
            attn_output, _ = self.multihead_attn(
                lstm_output, lstm_output, lstm_output
            )
            # Global average pooling
            return torch.mean(attn_output, dim=1)
        else:
            # Simple attention fallback
            scores = self.simple_attention(lstm_output).squeeze(-1)
            weights = F.softmax(scores, dim=1)
            return torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
            
class CNNLSTMBranchWithAttention(nn.Module):
    """Enhanced CNN-LSTM branch with attention mechanisms - FIXED VERSION"""
    
    def __init__(self, dropout_rate=0.3, use_attention=True, num_heads=None):
        super(CNNLSTMBranchWithAttention, self).__init__()
        self.use_attention = use_attention
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Attention mechanisms for CNN
        if self.use_attention:
            self.channel_attention = ChannelAttention(128)
            self.spatial_attention = SpatialAttention()
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=100, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=200, hidden_size=50, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Attention mechanism for LSTM - FIXED
        if self.use_attention:
            # 50*2 = 100 from bidirectional LSTM
            self.temporal_attention = TemporalAttention(100, num_heads=num_heads)
        
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
        x, _ = self.lstm1(x)  # (batch, 64, 200) - bidirectional
        x = self.lstm_dropout(x)
        
        x, _ = self.lstm2(x)  # (batch, 64, 100) - bidirectional
        x = self.lstm_dropout(x)
        
        # Apply temporal attention
        if self.use_attention:
            features, attention_weights = self.temporal_attention(x, use_multihead=False)
        else:
            # Fallback: use last timestep
            features = x[:, -1, :]  # (batch, 100)
        
        return features

class CNNLSTMIQModelWithAttention(nn.Module):
    """CNN-LSTM model with attention mechanisms for modulation classification - FIXED VERSION"""
    
    def __init__(self, num_classes, dropout_rate=0.4, use_attention=True, num_heads=None):
        super(CNNLSTMIQModelWithAttention, self).__init__()
        
        # Separate branches for I and Q signals
        self.i_branch = CNNLSTMBranchWithAttention(
            dropout_rate=dropout_rate * 0.75, 
            use_attention=use_attention,
            num_heads=num_heads
        )
        self.q_branch = CNNLSTMBranchWithAttention(
            dropout_rate=dropout_rate * 0.75, 
            use_attention=use_attention,
            num_heads=num_heads
        )
        
        # Feature fusion with attention
        feature_dim = 100  # From bidirectional LSTM (50*2)
        if use_attention:
            self.fusion_attention = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, 2),
                nn.Softmax(dim=1)
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
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
            i_input: I signal tensor of shape (batch, 1, 32, 32)
            q_input: Q signal tensor of shape (batch, 1, 32, 32)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Process I and Q signals separately
        i_features = self.i_branch(i_input)  # (batch, 100)
        q_features = self.q_branch(q_input)  # (batch, 100)
        
        # Feature fusion with attention
        if self.use_attention:
            # Concatenate I and Q features
            concat_features = torch.cat([i_features, q_features], dim=1)  # (batch, 200)
            
            # Learn attention weights for I and Q
            fusion_weights = self.fusion_attention(concat_features)  # (batch, 2)
            
            # Apply attention weights
            attended_i = i_features * fusion_weights[:, 0:1]
            attended_q = q_features * fusion_weights[:, 1:2]
            
            combined_features = attended_i + attended_q  # (batch, 100)
        else:
            # Simple addition fusion
            combined_features = i_features + q_features  # (batch, 100)
        
        # Classification
        output = self.classifier(combined_features)
        return output

def create_attention_CNNLSTMIQModel(n_labels=9, dropout_rate=0.5, use_attention=True, num_heads=None):  
    """
    Create CNN-LSTM model with attention mechanisms - FIXED VERSION
    
    Args:
        n_labels: Number of modulation classes
        dropout_rate: Dropout rate 
        use_attention: Whether to use attention mechanisms
        num_heads: Number of attention heads (auto-calculated if None)
    """
    return CNNLSTMIQModelWithAttention(
        num_classes=n_labels, 
        dropout_rate=dropout_rate, 
        use_attention=use_attention,
        num_heads=num_heads
    )