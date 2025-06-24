import torch
import torch.nn as nn
import torch.nn.functional as F

# Explicitly import gradient checkpointing if available
try:
    from torch.utils.checkpoint import checkpoint
except ImportError:
    checkpoint = None

class CNN_LSTM_Parallel(nn.Module):
    def __init__(self, 
                 input_channels=2,      # I/Q channels
                 sequence_length=1024,  # frame_size
                 num_classes=8,         # Reduced from 10 to 8 for memory efficiency
                 cnn_filters=[64, 128, 256],  # Reduced from [32, 64, 128] for 4GB VRAM
                 lstm_hidden_dim=128,    # Reduced from 64 to 32 for memory efficiency
                 lstm_num_layers=3,     # Reduced from 2 to 1 for memory efficiency
                 dropout=0.4,
                 use_checkpointing=True):  # Added gradient checkpointing option
        super(CNN_LSTM_Parallel, self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        self.use_checkpointing = use_checkpointing and (checkpoint is not None)

        # CNN Branch - Reduced kernels for memory efficiency
        self.cnn_kernels = [3, 5]  # Reduced from [3, 5, 7, 11] to save memory
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, cnn_filters[0], kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters[0]),
                nn.ReLU(inplace=True),  # Added inplace=True for memory efficiency
                nn.Dropout1d(0.1),
                
                nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters[1]),
                nn.ReLU(inplace=True),  # Added inplace=True
                nn.Dropout1d(0.1),
                
                nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters[2]),
                nn.ReLU(inplace=True),  # Added inplace=True
                nn.AdaptiveMaxPool1d(1)  # Global max pooling
            ) for k in self.cnn_kernels
        ])
        
        # LSTM Branch - Reduced for memory efficiency
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Changed to False to reduce memory usage
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate concatenated feature size - UPDATED FOR SMALLER MODEL
        cnn_feature_size = len(self.cnn_kernels) * cnn_filters[2]  # 2 * 64 = 128
        lstm_feature_size = lstm_hidden_dim  # 32 (no bidirectional)
        total_features = cnn_feature_size + lstm_feature_size  # 160
        
        # Simplified Classifier for memory efficiency
        # Using LayerNorm instead of BatchNorm to handle single-sample batches
        self.classifier = nn.Sequential(
            nn.Linear(total_features, total_features // 2),  # 160 -> 80
            nn.LayerNorm(total_features // 2),  # LayerNorm works with batch_size=1
            nn.ReLU(inplace=True),  # Added inplace=True
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, num_classes)  # 80 -> 8
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        # Input shape: (batch_size, 2, 1024)
        batch_size = x.size(0)
        
        # CNN Branch - With optional gradient checkpointing
        cnn_outputs = []
        for conv in self.convs:
            if self.use_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                # Following PyTorch docs: use_reentrant=False is recommended
                cnn_out = checkpoint(conv, x, use_reentrant=False)
            else:
                cnn_out = conv(x)  # (batch, filters, 1)
            cnn_out = cnn_out.squeeze(-1)  # (batch, filters)
            cnn_outputs.append(cnn_out)
        
        cnn_features = torch.cat(cnn_outputs, dim=1)  # (batch, total_cnn_features)
        cnn_features = self.dropout(cnn_features)
        
        # LSTM Branch
        lstm_input = x.transpose(1, 2)  # (batch, 1024, 2)
        
        if self.use_checkpointing and self.training:
            # Following PyTorch docs: use_reentrant=False is recommended
            lstm_out, (h_n, c_n) = checkpoint(self.lstm, lstm_input, use_reentrant=False)
        else:
            lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Use final hidden state (no bidirectional now)
        lstm_features = h_n[-1]  # (batch, hidden_dim)
        lstm_features = self.dropout(lstm_features)
        
        # Concatenate CNN and LSTM features
        combined_features = torch.cat([cnn_features, lstm_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        return output