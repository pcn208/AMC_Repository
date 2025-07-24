import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_1(nn.Module):
    """Shallow CNN branch from the diagram (CNN_1)"""
    def __init__(self, dropout_rate=0.3):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

class CNN_2(nn.Module):
    """Deeper CNN branch from the diagram (CNN_2)"""
    def __init__(self, dropout_rate=0.3):
        super(CNN_2, self).__init__()
        self.conv3 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

class DiagramModel(nn.Module):
    """Main model implementing the architecture from the diagram"""
    def __init__(self, num_classes, dropout_rate=0.4):
        super(DiagramModel, self).__init__()
        
        self.cnn1 = CNN_1(dropout_rate)
        self.cnn2 = CNN_2(dropout_rate)
        
        fused_feature_size = 128 + 256 # 384
        
        self.lstm1 = nn.LSTM(input_size=fused_feature_size, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        self.dense = nn.Linear(128, num_classes)

    def forward(self, x):
        features1 = self.cnn1(x)
        features2 = self.cnn2(x)
        
        fused_features = torch.cat((features1, features2), dim=1)
        
        lstm_input = fused_features.unsqueeze(1)
        
        lstm_out, _ = self.lstm1(lstm_input)
        lstm_out = self.lstm_dropout(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        
        last_time_step = lstm_out[:, -1, :]
        
        output = self.dense(last_time_step)
        
        return output

class FFTProcessor(nn.Module):
    """Handles FFT preprocessing of I/Q signals"""
    def __init__(self, H, W):
        super(FFTProcessor, self).__init__()
        self.H = H
        self.W = W
    
    def forward(self, i_signal, q_signal):
        """
        Convert I/Q signals to magnitude and phase spectrograms
        
        Args:
            i_signal: I component of the signal
            q_signal: Q component of the signal
            
        Returns:
            magnitude: Log magnitude spectrum
            phase: Phase spectrum
        """
        # Create complex signal
        complex_sig = torch.complex(i_signal, q_signal)
        
        # Apply FFT
        fft_res = torch.fft.fft(complex_sig)
        
        # Extract magnitude and phase
        magnitude = torch.abs(fft_res)
        phase = torch.angle(fft_res)
        
        # Apply log transformation for better dynamic range
        log_magnitude = torch.log1p(magnitude)  # log(1 + magnitude) to avoid log(0)
        
        # Normalize
        log_magnitude = (log_magnitude - log_magnitude.mean()) / (log_magnitude.std() + 1e-8)
        phase = (phase - phase.mean()) / (phase.std() + 1e-8)
        
        # Reshape to 2D for CNN processing
        mag_2d = log_magnitude.view(-1, 1, self.H, self.W)
        phase_2d = phase.view(-1, 1, self.H, self.W)
        
        return mag_2d, phase_2d

class DiagramFFTModel(nn.Module):
    """Enhanced model using FFT-processed magnitude and phase"""
    def __init__(self, num_classes, H, W, dropout_rate=0.5):
        super(DiagramFFTModel, self).__init__()
        
        # FFT preprocessing
        self.fft_processor = FFTProcessor(H, W)
        
        # Separate branches for magnitude and phase
        self.magnitude_branch = DiagramModel(num_classes, dropout_rate)
        self.phase_branch = DiagramModel(num_classes, dropout_rate)
        
        # Optional: Combined processing branch
        self.use_combined_branch = True
        if self.use_combined_branch:
            self.combined_branch = DiagramModel(num_classes, dropout_rate)
        
    def forward(self, i_input, q_input):
        # Convert I/Q to magnitude and phase
        magnitude, phase = self.fft_processor(i_input, q_input)
        
        # Process magnitude and phase separately
        mag_logits = self.magnitude_branch(magnitude)
        phase_logits = self.phase_branch(phase)
        
        if self.use_combined_branch:
            # Option 1: Concatenate magnitude and phase channels
            combined_input = torch.cat([magnitude, phase], dim=1)  # Shape: (batch, 2, H, W)
            
            # For this, you'd need to modify CNN_1 and CNN_2 to accept 2 input channels
            # For now, let's use a simple approach: process them separately and combine
            combined_logits = (mag_logits + phase_logits) / 2
            
            return {
                'magnitude': mag_logits,
                'phase': phase_logits, 
                'combined': combined_logits
            }
        else:
            # Simple averaging approach
            combined_logits = (mag_logits + phase_logits) / 2
            return combined_logits

# Alternative: Enhanced CNN that can handle 2-channel input (magnitude + phase together)
class Enhanced_CNN_1(nn.Module):
    """CNN that can process magnitude and phase together"""
    def __init__(self, input_channels=2, dropout_rate=0.3):
        super(Enhanced_CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

class DiagramFFTCombinedModel(nn.Module):
    """Model that processes magnitude and phase as 2-channel input"""
    def __init__(self, num_classes, H, W, dropout_rate=0.5):
        super(DiagramFFTCombinedModel, self).__init__()
        
        self.fft_processor = FFTProcessor(H, W)
        
        # Enhanced CNNs that accept 2-channel input
        self.cnn1 = Enhanced_CNN_1(input_channels=2, dropout_rate=dropout_rate)
        self.cnn2 = Enhanced_CNN_1(input_channels=2, dropout_rate=dropout_rate)  # You'd create Enhanced_CNN_2 similarly
        
        fused_feature_size = 128 + 128  # Adjust based on your Enhanced_CNN_2 output
        
        self.lstm1 = nn.LSTM(input_size=fused_feature_size, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        self.dense = nn.Linear(128, num_classes)
    
    def forward(self, i_input, q_input):
        # Convert I/Q to magnitude and phase
        magnitude, phase = self.fft_processor(i_input, q_input)
        
        # Combine magnitude and phase as 2-channel input
        combined_input = torch.cat([magnitude, phase], dim=1)  # Shape: (batch, 2, H, W)
        
        # Process through CNN branches
        features1 = self.cnn1(combined_input)
        features2 = self.cnn1(combined_input)  # Using same CNN for simplicity
        
        fused_features = torch.cat((features1, features2), dim=1)
        
        # LSTM processing
        lstm_input = fused_features.unsqueeze(1)
        lstm_out, _ = self.lstm1(lstm_input)
        lstm_out = self.lstm_dropout(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        
        last_time_step = lstm_out[:, -1, :]
        output = self.dense(last_time_step)
        
        return output

# Factory functions
def create_diagram_fft_model(n_labels=9, H=64, W=64, dropout_rate=0.5):
    """Creates the FFT-enhanced model with separate magnitude/phase processing"""
    return DiagramFFTModel(num_classes=n_labels, H=H, W=W, dropout_rate=dropout_rate)

def create_diagram_fft_combined_model(n_labels=9, H=64, W=64, dropout_rate=0.5):
    """Creates the FFT-enhanced model with combined magnitude/phase processing"""
    return DiagramFFTCombinedModel(num_classes=n_labels, H=H, W=W, dropout_rate=dropout_rate)