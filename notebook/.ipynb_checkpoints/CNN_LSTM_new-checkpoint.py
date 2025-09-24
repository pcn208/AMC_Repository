import torch
import torch.nn as nn
import torch.nn.functional as F  # Corrected import

class Cnn2DBranch(nn.Module): 
    def __init__(self, in_channels=1, dropout_rate=0.4): 
        super(Cnn2DBranch, self).__init__() 

        self.backbone = nn.Sequential(
            # Block1 
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Added missing layer
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.1, inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout2d(dropout_rate), 
            
            # Block2 
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout2d(dropout_rate) 
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.output_features = 256  # Number of channels from the last CNN layer 

    def forward(self, x): 
        # x shape: (batch, 1, 32, 32) 
        feature_map = self.backbone(x) 
        pooled = self.global_pool(feature_map) 
        features = pooled.view(pooled.size(0), -1)  # Fixed variable name
        return features 

class Lstm1dBranch(nn.Module): 
    def __init__(self, in_channels=2, dropout_rate=0.4): 
        super(Lstm1dBranch, self).__init__()  # Added missing super() call

        self.preprocessor = nn.Sequential(
            # Block 1 
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),  # Fixed parameter order
            nn.BatchNorm1d(64),
            nn.ReLU(),  # Fixed typo: RelU -> ReLU
            nn.MaxPool1d(kernel_size=4),
            
            # Block 2 
            nn.Conv1d(64, 128, kernel_size=7, padding=3), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=4)
        )
        
        # LSTM to process the higher-level, shorter sequence 
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout_rate
        )
        
        self.output_features = 256 * 2  # Fixed typo: ouput -> output

    def forward(self, x): 
        # x shape: (batch, 1024, 2) 
        # Conv1d expects (batch, channels, seq_len) 
        x = x.permute(0, 2, 1) 

        # Pass through pre-processor 
        x = self.preprocessor(x) 
        
        # Reshape for LSTM (batch, seq_len, features) 
        x = x.permute(0, 2, 1) 

        self.lstm.flatten_parameters() 
        _, (h_n, _) = self.lstm(x) 

        # Concatenate the final forward and backward hidden states 
        # h_n shape is (num_layers * num_directions, batch, hidden_size) 
        features = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) 

        return features 

class MultiDomainFusionModel(nn.Module): 
    def __init__(self, num_classes, dropout_rate=0.5): 
        super(MultiDomainFusionModel, self).__init__() 

        self.amplitude_cnn_branch = Cnn2DBranch(dropout_rate=dropout_rate) 
        self.phase_cnn_branch = Cnn2DBranch(dropout_rate=dropout_rate) 
        self.iq_lstm_branch = Lstm1dBranch(dropout_rate=dropout_rate) 

        # Fusion and classifier 
        total_features = (self.amplitude_cnn_branch.output_features + 
                          self.phase_cnn_branch.output_features +
                          self.iq_lstm_branch.output_features)  # Fixed typo
        
        self.classifier = nn.Sequential(  # Fixed typo: squential -> Sequential
            nn.Linear(total_features, 512), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Fixed typo: dropot -> dropout
            nn.Linear(256, num_classes)
        )

    def forward(self, amplitude_input, phase_input, iq_sequence): 
        # Process each input through its dedicated expert branch 
        amp_features = self.amplitude_cnn_branch(amplitude_input) 
        phase_features = self.phase_cnn_branch(phase_input) 
        iq_features = self.iq_lstm_branch(iq_sequence) 

        # Fuse the insights from all experts by concatenating their feature vectors
        fused_features = torch.cat([amp_features, phase_features, iq_features], dim=1) 
        
        # Make the final classification 
        output = self.classifier(fused_features)  # Fixed variable name
        return output 

# Fixed function definition
def create_multi_domain_model(num_classes, dropout_rate=0.5): 
    model = MultiDomainFusionModel(num_classes=num_classes, dropout_rate=dropout_rate) 
    return model
