import torch
import torch.nn as nn

class BaseCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.block(x)

class BaseCNN_NET(nn.Module):
    def __init__(self, n_labels, dropout_rate=0.4):
        super().__init__()
        self.backbone = nn.Sequential(
            BaseCNN_Block(2, 32, dropout_rate=0.2),
            BaseCNN_Block(32, 64, dropout_rate=0.3),
            BaseCNN_Block(64, 128, dropout_rate=0.3),
            nn.AdaptiveAvgPool1d(4)
        )
        
        # FIXED: Correct input features calculation
        # With your input (2, 1024) and the architecture above,
        # the backbone outputs (128, 4) -> 512 features after flattening
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),  # CORRECTED: 512 not 1024
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(64, n_labels)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

def create_BaseCNN_NET(n_labels=6, dropout_rate=0.5):  
    return BaseCNN_NET(n_labels, dropout_rate)