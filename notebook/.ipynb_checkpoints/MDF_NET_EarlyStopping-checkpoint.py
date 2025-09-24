import torch
import torch.nn as nn
import torch.nn.functional as F

# Option 1: Input-Level Early Fusion for Spectrograms
class InputFusionModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(InputFusionModel, self).__init__()
        
        # Fuse amplitude and phase at input level (2 channels)
        self.spectrogram_branch = Cnn2DBranch(in_channels=2, dropout_rate=dropout_rate)
        self.iq_lstm_branch = Lstm1dBranch(dropout_rate=dropout_rate)
        
        # Smaller fusion layer since we have fewer branches
        total_features = self.spectrogram_branch.output_features + self.iq_lstm_branch.output_features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, amplitude_input, phase_input, iq_sequence):
        # Convert inputs to float32 to match model weights (for mixed precision compatibility)
        amplitude_input = amplitude_input.float()
        phase_input = phase_input.float()
        iq_sequence = iq_sequence.float()
        
        # Combine amplitude and phase spectrograms
        combined_spectrogram = torch.cat([amplitude_input, phase_input], dim=1)
        
        # Process combined spectrogram and I/Q separately
        spectrogram_features = self.spectrogram_branch(combined_spectrogram)
        iq_features = self.iq_lstm_branch(iq_sequence)
        
        # Final fusion
        fused_features = torch.cat([spectrogram_features, iq_features], dim=1)
        output = self.classifier(fused_features)
        return output

# Option 2: Shared Backbone with Branch Specialization
class SharedBackboneModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(SharedBackboneModel, self).__init__()
        
        # Shared early layers for spectrograms
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Specialized branches
        self.amp_branch = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.phase_branch = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.iq_lstm_branch = Lstm1dBranch(dropout_rate=dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        total_features = 256 + 256 + self.iq_lstm_branch.output_features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, amplitude_input, phase_input, iq_sequence):
        # Convert inputs to float32 to match model weights (for mixed precision compatibility)
        amplitude_input = amplitude_input.float()
        phase_input = phase_input.float()
        iq_sequence = iq_sequence.float()
        
        # Shared processing for both spectrograms
        amp_shared = self.shared_backbone(amplitude_input)
        phase_shared = self.shared_backbone(phase_input)
        
        # Branch-specific processing
        amp_features = self.amp_branch(amp_shared)
        phase_features = self.phase_branch(phase_shared)
        
        # Global pooling
        amp_pooled = self.global_pool(amp_features).view(amp_features.size(0), -1)
        phase_pooled = self.global_pool(phase_features).view(phase_features.size(0), -1)
        
        # I/Q processing (unchanged)
        iq_features = self.iq_lstm_branch(iq_sequence)
        
        # Final fusion
        fused_features = torch.cat([amp_pooled, phase_pooled, iq_features], dim=1)
        output = self.classifier(fused_features)
        return output

# Option 3: Cross-Modal Attention Fusion
class AttentionFusionModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(AttentionFusionModel, self).__init__()
        
        self.amplitude_cnn_branch = Cnn2DBranch(dropout_rate=dropout_rate)
        self.phase_cnn_branch = Cnn2DBranch(dropout_rate=dropout_rate)
        self.iq_lstm_branch = Lstm1dBranch(dropout_rate=dropout_rate)
        
        # Cross-attention between amplitude and phase features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature dimension matching for attention
        self.feature_proj = nn.Linear(256, 256)
        
        total_features = 256 + 256 + self.iq_lstm_branch.output_features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, amplitude_input, phase_input, iq_sequence):
        # Convert inputs to float32 to match model weights
        amplitude_input = amplitude_input.float()
        phase_input = phase_input.float()
        iq_sequence = iq_sequence.float()
        
        # Extract features from each branch
        amp_features = self.amplitude_cnn_branch(amplitude_input)
        phase_features = self.phase_cnn_branch(phase_input)
        iq_features = self.iq_lstm_branch(iq_sequence)
        
        # Project features for attention (add sequence dimension)
        amp_proj = self.feature_proj(amp_features).unsqueeze(1)  # (batch, 1, 256)
        phase_proj = self.feature_proj(phase_features).unsqueeze(1)  # (batch, 1, 256)
        
        # Cross-attention between amplitude and phase
        amp_attended, _ = self.cross_attention(amp_proj, phase_proj, phase_proj)
        phase_attended, _ = self.cross_attention(phase_proj, amp_proj, amp_proj)
        
        # Remove sequence dimension
        amp_attended = amp_attended.squeeze(1)
        phase_attended = phase_attended.squeeze(1)
        
        # Final fusion
        fused_features = torch.cat([amp_attended, phase_attended, iq_features], dim=1)
        output = self.classifier(fused_features)
        return output

# Option 4: Progressive Fusion with Feature Interaction
class ProgressiveFusionModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ProgressiveFusionModel, self).__init__()
        
        # Early processing
        self.amp_early = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.phase_early = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Fusion layer for early features
        self.early_fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),  # 1x1 conv to mix channels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Later processing on fused features
        self.later_processing = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.iq_lstm_branch = Lstm1dBranch(dropout_rate=dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        total_features = 256 + self.iq_lstm_branch.output_features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, amplitude_input, phase_input, iq_sequence):
        # Convert inputs to float32 to match model weights (for mixed precision compatibility)
        amplitude_input = amplitude_input.float()
        phase_input = phase_input.float()
        iq_sequence = iq_sequence.float()
        
        # Early separate processing
        amp_early = self.amp_early(amplitude_input)
        phase_early = self.phase_early(phase_input)
        
        # Early fusion
        combined_early = torch.cat([amp_early, phase_early], dim=1)
        fused_early = self.early_fusion(combined_early)
        
        # Later processing on fused features
        fused_features = self.later_processing(fused_early)
        fused_pooled = self.global_pool(fused_features).view(fused_features.size(0), -1)
        
        # I/Q processing (unchanged)
        iq_features = self.iq_lstm_branch(iq_sequence)
        
        # Final fusion
        final_features = torch.cat([fused_pooled, iq_features], dim=1)
        output = self.classifier(final_features)
        return output

# Model creation function
def create_model(model_type, num_classes, dropout_rate=0.5):
    """
    Create different fusion model types.
    
    Args:
        model_type (str): 'input', 'shared', 'attention', 'progressive', or 'multi_domain'
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        torch.nn.Module: The requested fusion model
    """
    models = {
        'input': InputFusionModel,
        'shared': SharedBackboneModel,
        'attention': AttentionFusionModel,
        'progressive': ProgressiveFusionModel,
        'multi_domain': MultiDomainFusionModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: {list(models.keys())}")
    
    return models[model_type](num_classes, dropout_rate)

# Original late fusion model from your code
class MultiDomainFusionModel(nn.Module): 
    def __init__(self, num_classes, dropout_rate=0.5): 
        super(MultiDomainFusionModel, self).__init__() 

        self.amplitude_cnn_branch = Cnn2DBranch(dropout_rate=dropout_rate) 
        self.phase_cnn_branch = Cnn2DBranch(dropout_rate=dropout_rate) 
        self.iq_lstm_branch = Lstm1dBranch(dropout_rate=dropout_rate) 

        total_features = (self.amplitude_cnn_branch.output_features + 
                          self.phase_cnn_branch.output_features +
                          self.iq_lstm_branch.output_features)
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, amplitude_input, phase_input, iq_sequence): 
        # Convert inputs to float32 to match model weights (for mixed precision compatibility)
        amplitude_input = amplitude_input.float()
        phase_input = phase_input.float()
        iq_sequence = iq_sequence.float()
        
        amp_features = self.amplitude_cnn_branch(amplitude_input) 
        phase_features = self.phase_cnn_branch(phase_input) 
        iq_features = self.iq_lstm_branch(iq_sequence) 

        fused_features = torch.cat([amp_features, phase_features, iq_features], dim=1) 
        output = self.classifier(fused_features)
        return output

# Keep your original branch classes
class Cnn2DBranch(nn.Module): 
    def __init__(self, in_channels=1, dropout_rate=0.4): 
        super(Cnn2DBranch, self).__init__() 

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.1, inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout2d(dropout_rate), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout2d(dropout_rate) 
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.output_features = 256

    def forward(self, x): 
        feature_map = self.backbone(x) 
        pooled = self.global_pool(feature_map) 
        features = pooled.view(pooled.size(0), -1)
        return features 

class Lstm1dBranch(nn.Module): 
    def __init__(self, in_channels=2, dropout_rate=0.4): 
        super(Lstm1dBranch, self).__init__()

        self.preprocessor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            
            nn.Conv1d(64, 128, kernel_size=7, padding=3), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=4)
        )
        
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout_rate
        )
        
        self.output_features = 256 * 2

    def forward(self, x): 
        x = x.permute(0, 2, 1) 
        x = self.preprocessor(x) 
        x = x.permute(0, 2, 1) 

        self.lstm.flatten_parameters() 
        _, (h_n, _) = self.lstm(x) 
        features = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) 
        return features