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

class DiagramIQModel(nn.Module):
    """Top-level model for I/Q signals using the DiagramModel architecture"""
    def __init__(self, num_classes, dropout_rate=0.5):
        super(DiagramIQModel, self).__init__()
        self.i_branch = DiagramModel(num_classes, dropout_rate)
        self.q_branch = DiagramModel(num_classes, dropout_rate)
        
    def forward(self, i_input, q_input):
        i_logits = self.i_branch(i_input)
        q_logits = self.q_branch(q_input)
        
        combined_logits = (i_logits + q_logits) / 2
        return combined_logits

# --- ADDED FACTORY FUNCTION ---
def create_diagram_iq_model(n_labels=9, dropout_rate=0.5):
    """
    Creates the DiagramIQModel instance.
    
    Args:
        n_labels: Number of output classes.
        dropout_rate: The base dropout rate for the model.
    
    Returns:
        An instance of DiagramIQModel.
    """
    return DiagramIQModel(num_classes=n_labels, dropout_rate=dropout_rate)