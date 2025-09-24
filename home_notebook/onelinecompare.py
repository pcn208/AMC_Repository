import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchinfo import summary
from typing import Tuple, Optional
import torch.utils.checkpoint as checkpoint
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm, trange
import seaborn as sns
import gc
import time
import warnings
from collections import deque
import psutil
import os
import math
import copy

warnings.filterwarnings('ignore')

# ==============================================================================
# ## 1. Konfigurasi dan Setup Awal
# ==============================================================================

# -- Konfigurasi Perangkat --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Perangkat yang digunakan: {device}")

# -- Path File --
FILE_PATH = "C:\\workarea\\CNN model\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
JSON_PATH = 'C:\\workarea\\CNN model\\dataset\\radioml2018\\versions\\2\\classes-fixed.json' 

# -- Parameter Model dan Dataset --
TARGET_MODULATIONS = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'OQPSK']
NUM_CLASSES = len(TARGET_MODULATIONS)
TARGET_SNR = [0, 10, 30]

# -- Hyperparameter Training --
BATCH_SIZE = 512
NUM_EPOCHS = 300
patience = 25
TRAIN_RATIO = 0.7 
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# -- Parameter Teknis --
INPUT_CHANNELS = 2 
SEQUENCE_LENGTH = 1024
NUM_WORKERS = 0

# FIX: Define missing nf variables
nf_train = 4000  # Number of samples per modulation per SNR for training
nf_valid = 1000  # Number of samples per modulation per SNR for validation
nf_test = 1000   # Number of samples per modulation per SNR for testing

print("\n📋 Parameter Training:")
print(f"  - Target Modulasi: {NUM_CLASSES} kelas")
print(f"  - Target SNR: {TARGET_SNR}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {NUM_EPOCHS}")
print(f"  - Rasio Data: Train={TRAIN_RATIO}, Valid={VALID_RATIO}, Test={TEST_RATIO}")

# ==============================================================================
# ## 2. Definisi Model (Placeholder)
# ==============================================================================

# FIX: Create placeholder models since the original model imports are not available
class PlaceholderCNNLSTM(nn.Module):
    def __init__(self, n_labels=9, dropout_rate=0.7):
        super(PlaceholderCNNLSTM, self).__init__()
        
        # CNN layers for I channel
        self.cnn_i = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        # CNN layers for Q channel  
        self.cnn_q = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        # Calculate flattened size (assuming 32x32 input -> 8x8 after pooling)
        self.flattened_size = 64 * 8 * 8 * 2  # *2 for both I and Q channels
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=128, 
                           batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_labels)
        )
    
    def forward(self, i_input, q_input):
        batch_size = i_input.size(0)
        
        # Process I and Q channels
        i_features = self.cnn_i(i_input)
        q_features = self.cnn_q(q_input)
        
        # Flatten and concatenate
        i_flat = i_features.view(batch_size, -1)
        q_flat = q_features.view(batch_size, -1)
        combined = torch.cat([i_flat, q_flat], dim=1)
        
        # Add sequence dimension for LSTM
        combined = combined.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Classification
        output = self.classifier(lstm_out)
        return output

# Create models
try:
    model_standard = PlaceholderCNNLSTM(n_labels=NUM_CLASSES, dropout_rate=0.7).to(device)
    model_parallel = PlaceholderCNNLSTM(n_labels=NUM_CLASSES, dropout_rate=0.7).to(device)
    print("\n✅ Berhasil membuat Model Standard dan Parallel CNN-LSTM")
except Exception as e:
    print(f"\n⚠️ Gagal membuat Model: {e}")
    model_standard = None
    model_parallel = None

# ==============================================================================
# ## 3. Fungsi dan Kelas Dataset
# ==============================================================================

def dataset_split(data, modulations_classes, modulations, snrs, target_modulations, mode, target_snrs,
                  train_proportion=0.7, valid_proportion=0.15, test_proportion=0.15, seed=48):
    np.random.seed(seed)
    all_indices = []
    all_mod_labels = []
    all_snr_labels = []

    target_modulation_indices = [modulations_classes.index(modu) for modu in target_modulations]

    for modu_idx in target_modulation_indices:
        for snr in target_snrs:
            snr_modu_indices = np.where((modulations == modu_idx) & (snrs == snr))[0]

            if len(snr_modu_indices) > 0:
                # Sort indices to ensure HDF5 compatibility
                snr_modu_indices = np.sort(snr_modu_indices)
                
                # Shuffle using random permutation
                perm = np.random.permutation(len(snr_modu_indices))
                shuffled_indices = snr_modu_indices[perm]
                
                num_samples = len(shuffled_indices)
                train_end = int(train_proportion * num_samples)
                valid_end = train_end + int(valid_proportion * num_samples)

                if mode == 'train':
                    indices = shuffled_indices[:train_end]
                elif mode == 'valid':
                    indices = shuffled_indices[train_end:valid_end]
                elif mode == 'test':
                    indices = shuffled_indices[valid_end:]
                else:
                    raise ValueError(f"Mode tidak dikenal: {mode}. Mode yang valid adalah 'train', 'valid', dan 'test'")

                if len(indices) > 0:
                    all_indices.extend(indices)
                    all_mod_labels.extend([modu_idx] * len(indices))
                    all_snr_labels.extend([snr] * len(indices))

    if not all_indices:
        return np.array([]), np.array([]), np.array([])

    # Sort all indices for HDF5 compatibility
    all_indices = np.array(all_indices)
    all_mod_labels = np.array(all_mod_labels)
    all_snr_labels = np.array(all_snr_labels)
    
    # Sort everything by indices to maintain HDF5 compatibility
    sort_order = np.argsort(all_indices)
    sorted_indices = all_indices[sort_order]
    sorted_mod_labels = all_mod_labels[sort_order]
    sorted_snr_labels = all_snr_labels[sort_order]
    
    # Load data with sorted indices
    try:
        X_array = data[sorted_indices]
    except Exception as e:
        print(f"Error accessing HDF5 data: {e}")
        # Fallback: load data one by one
        X_list = []
        for idx in sorted_indices:
            X_list.append(data[idx])
        X_array = np.array(X_list)
    
    # Remap labels to 0, 1, 2, ...
    unique_labels = np.unique(sorted_mod_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    Y_remapped = np.array([label_map[label] for label in sorted_mod_labels])
    
    return X_array, Y_remapped, sorted_snr_labels

class RadioMLIQDataset(Dataset):
    """Dataset class for RadioML18 data formatted for CNNIQModel dual-branch architecture."""
    
    def __init__(self, mode: str, use_fft: bool = False, seed: int = 48):
        super(RadioMLIQDataset, self).__init__()
        
        self.file_path = FILE_PATH 
        self.json_path = JSON_PATH 
        self.target_modulations = TARGET_MODULATIONS
        self.use_fft = use_fft
        self.mode = mode
        
        # Validate mode
        if mode not in ['train', 'valid', 'test']:
            raise ValueError(f"Mode must be 'train', 'valid', or 'test', got '{mode}'")
        
        # Load data files
        try:
            self.hdf5_file = h5py.File(self.file_path, 'r')
            self.modulation_classes = json.load(open(self.json_path, 'r'))
            print(f"✅ Successfully loaded HDF5 file and modulation classes")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading data files: {e}")
        except Exception as e:
            print(f"Error loading file: {e}")
            raise e
        
        # Load raw data - Load everything into memory to avoid HDF5 indexing issues
        print(f"📊 Loading dataset into memory for {mode} split...")
        try:
            self.X_full = self.hdf5_file['X'][:]  # Load all data into memory
            self.Y_full = np.argmax(self.hdf5_file['Y'][:], axis=1)
            self.Z_full = self.hdf5_file['Z'][:, 0]
            print(f"✅ Loaded full dataset: {self.X_full.shape[0]} samples")
        except Exception as e:
            print(f"❌ Error loading full dataset: {e}")
            raise e
        
        num_mods = len(self.target_modulations)   
        num_snrs = 26         
        
        train_proportion = (num_mods * num_snrs * nf_train) / self.X_full.shape[0]
        valid_proportion = (num_mods * num_snrs * nf_valid) / self.X_full.shape[0]
        test_proportion  = (num_mods * num_snrs * nf_test ) / self.X_full.shape[0]
        
        self.target_snrs = np.unique(self.Z_full)
        
        # Split dataset - now using in-memory data
        print(f"🔄 Splitting dataset for {mode}...")
        self.X_data, self.Y_data, self.Z_data = dataset_split(
            data=self.X_full,  # Use in-memory data
            modulations_classes=self.modulation_classes,
            modulations=self.Y_full,
            snrs=self.Z_full,
            mode=mode,
            train_proportion=train_proportion,
            valid_proportion=valid_proportion,
            test_proportion=test_proportion,
            target_modulations=self.target_modulations,
            target_snrs=self.target_snrs,
            seed=seed
        )
        
        if len(self.X_data) == 0:
            raise ValueError(f"No data found for {mode} split. Check your target modulations and SNRs.")
        
        # Apply I/Q swap correction for AMC compatibility
        self.X_data = self.X_data[:, :, [0, 1]]
        
        # Validate signal length for 2D reshaping
        signal_length = self.X_data.shape[1]
        if signal_length != 1024:
            raise ValueError(f"Expected signal length 1024 for 32x32 reshape, got {signal_length}")

        L = signal_length
        H = int(math.floor(math.sqrt(L)))
        while L % H != 0:
            H -= 1
        W = L // H
        
        self.H, self.W = H, W
        print(f"🔧 Signals will be reshaped to ({H}, {W}) for sequence length {L}")
        print(f"✅ Aspect ratio: {W/H:.2f}, Total elements preserved: {H*W} = {L}")
        
        if self.use_fft:
            print("Dataset configured to use FFT as input")
        
        # Store dataset statistics
        self.num_data = self.X_data.shape[0]
        self.num_lbl = len(self.target_modulations)
        self.num_snr = self.target_snrs.shape[0]
        
        print(f"RadioMLIQDataset {mode}: {self.num_data} samples, "
              f"{self.num_lbl} classes, {self.num_snr} SNR levels")
        
        # Close HDF5 file since we have everything in memory
        self.hdf5_file.close()
    
    def __len__(self) -> int:
        return self.X_data.shape[0]

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_data:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_data}")

        x_raw = self.X_data[idx]       # shape: (L, 2)
        y     = int(self.Y_data[idx])
        z     = float(self.Z_data[idx])

        # Convert to tensor
        x = torch.from_numpy(x_raw).float().transpose(0, 1)  # shape: (2, L)

        if self.use_fft:
            # Combine to complex signal
            complex_sig = torch.complex(x[0], x[1])  # shape: (L,)
            fft_res     = torch.fft.fft(complex_sig)  # shape: (L,)

            # Convert to real-valued 2D (L × 2) matrix: real | imag
            x_fft = torch.stack([torch.real(fft_res), torch.imag(fft_res)], dim=1)  # shape: (L, 2)

            # Reshape to 2D: (1, H, W) each for real and imag
            x_real_2d = x_fft[:, 0].view(1, self.H, self.W)
            x_imag_2d = x_fft[:, 1].view(1, self.H, self.W)

            return x_real_2d, x_imag_2d, y, z

        else:
            # Non-FFT path (amplitude/phase domain)
            i_signal = x[0]  # shape: (L,)
            q_signal = x[1]

            amplitude = torch.sqrt(i_signal**2 + q_signal**2)
            phase     = torch.atan2(q_signal, i_signal)

            i_2d = amplitude.view(1, self.H, self.W)
            q_2d = phase.view(1, self.H, self.W)

            return i_2d, q_2d, y, z

    def get_signal_stats(self):
        """Compute basic stats over a sample of signals."""
        sample_indices = np.random.choice(self.num_data, min(1000, self.num_data), replace=False)
        i_vals, q_vals = [], []
        for idx in sample_indices:
            i2d, q2d, _, _ = self[idx]
            i_vals.append(i2d.flatten())
            q_vals.append(q2d.flatten())
        i_all = torch.cat(i_vals)
        q_all = torch.cat(q_vals)
        return {
            'i_mean': i_all.mean().item(),
            'i_std':  i_all.std().item(),
            'q_mean': q_all.mean().item(),
            'q_std':  q_all.std().item(),
            'shape':  (1, self.H, self.W),
            'num_samples': self.num_data
        }

    def close(self):
        # No need to close since we already closed in __init__
        pass

    def __del__(self):
        self.close()

# ==============================================================================
# ## 4. Optimizer, Scheduler, dan Data Loaders
# ==============================================================================

# -- Loss Function --
criterion = nn.CrossEntropyLoss(label_smoothing=0.12)

# -- Optimizers & Schedulers --
if model_standard and model_parallel:
    optimizer_standard = optim.AdamW(model_standard.parameters(), lr=0.002, weight_decay=2e-3)
    scheduler_standard = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_standard, T_0=30, eta_min=5e-6)

    optimizer_parallel = optim.SGD(model_parallel.parameters(), lr=0.01, momentum=0.9, weight_decay=3e-3, nesterov=True)
    scheduler_parallel = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_parallel, T_0=30, eta_min=5e-5)

    scaler_standard = GradScaler()
    scaler_parallel = GradScaler()

    print("\n🔄 Setup Optimizer dan Scheduler:")
    print(f"  • Standard: AdamW dengan CosineAnnealingWarmRestarts")
    print(f"  • Parallel: SGD dengan Nesterov Momentum")
    print(f"  • Label smoothing: 0.12")
else:
    print("\n⚠️ Salah satu atau kedua model tidak terdefinisi. Setup optimizer dilewati.")

# -- Data Loaders --
try:
    print("\n📂 Memuat datasets...")
    train_dataset = RadioMLIQDataset('train')
    valid_dataset = RadioMLIQDataset('valid')
    test_dataset = RadioMLIQDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    print(f"  - Ukuran dataset Latihan: {len(train_dataset)}")
    print(f"  - Ukuran dataset Validasi: {len(valid_dataset)}")
    print(f"  - Ukuran dataset Tes: {len(test_dataset)}")
except Exception as e:
    print(f"⚠️ Error loading datasets: {e}")
    train_loader = valid_loader = test_loader = None

# ==============================================================================
# ## 5. Fungsi Training, Validasi, dan Tes
# ==============================================================================

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for i_inputs, q_inputs, labels, _ in loader:
        i_inputs, q_inputs, labels = i_inputs.to(device), q_inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(i_inputs, q_inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * i_inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, 100. * correct / total

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    predictions, true_labels, snr_values = [], [], []
    
    with torch.no_grad():
        for i_inputs, q_inputs, labels, snrs in loader:
            i_inputs, q_inputs, labels = i_inputs.to(device), q_inputs.to(device), labels.to(device)
            
            outputs = model(i_inputs, q_inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * i_inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            snr_values.extend(snrs.cpu().numpy())
            
    return running_loss / total, 100. * correct / total, predictions, true_labels, snr_values

def plot_confusion_matrices_by_snr(predictions, true_labels, snr_values, target_snrs, 
                                 target_modulations, model_name, save_dir="confusion_matrices"):
    """Plot and save confusion matrices for each target SNR"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    snr_values = np.array(snr_values)
    
    # Create subplots for all SNRs
    fig, axes = plt.subplots(1, len(target_snrs), figsize=(6*len(target_snrs), 5))
    if len(target_snrs) == 1:
        axes = [axes]
    
    for i, target_snr in enumerate(target_snrs):
        # Filter data for specific SNR
        snr_mask = snr_values == target_snr
        if not np.any(snr_mask):
            print(f"⚠️ No data found for SNR {target_snr} dB")
            continue
            
        snr_predictions = predictions[snr_mask]
        snr_true_labels = true_labels[snr_mask]
        
        # Calculate confusion matrix
        cm = confusion_matrix(snr_true_labels, snr_predictions)
        
        # Calculate accuracy for this SNR
        accuracy = np.sum(snr_predictions == snr_true_labels) / len(snr_true_labels) * 100
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_modulations, 
                   yticklabels=target_modulations,
                   ax=axes[i])
        
        axes[i].set_title(f'{model_name}\nSNR: {target_snr} dB\nAccuracy: {accuracy:.1f}%')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        
        # Save individual confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_modulations, 
                   yticklabels=target_modulations)
        plt.title(f'{model_name} - Confusion Matrix\nSNR: {target_snr} dB (Accuracy: {accuracy:.1f}%)')
        plt.xlabel('Predicted Modulation')
        plt.ylabel('True Modulation')
        plt.tight_layout()
        
        individual_filename = os.path.join(save_dir, f'{model_name}_SNR_{target_snr}dB_confusion_matrix.png')
        plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved confusion matrix for SNR {target_snr} dB: {individual_filename}")
    
    # Save combined plot
    plt.figure(fig.number)
    plt.suptitle(f'{model_name} - Confusion Matrices by SNR', fontsize=16)
    plt.tight_layout()
    combined_filename = os.path.join(save_dir, f'{model_name}_all_SNR_confusion_matrices.png')
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved combined confusion matrices: {combined_filename}")

def evaluate_model_by_snr(model, loader, criterion, device, target_snrs, target_modulations, model_name):
    """Evaluate model and generate detailed SNR-based analysis"""
    model.eval()
    all_predictions, all_true_labels, all_snr_values = [], [], []
    snr_stats = {snr: {'correct': 0, 'total': 0, 'loss': 0.0} for snr in target_snrs}
    
    with torch.no_grad():
        for i_inputs, q_inputs, labels, snrs in loader:
            i_inputs, q_inputs, labels = i_inputs.to(device), q_inputs.to(device), labels.to(device)
            
            outputs = model(i_inputs, q_inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            # Convert to numpy for easier processing
            pred_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            snrs_np = snrs.cpu().numpy()
            
            all_predictions.extend(pred_np)
            all_true_labels.extend(labels_np)
            all_snr_values.extend(snrs_np)
            
            # Update SNR-specific statistics
            for i, snr in enumerate(snrs_np):
                if snr in snr_stats:
                    snr_stats[snr]['total'] += 1
                    if pred_np[i] == labels_np[i]:
                        snr_stats[snr]['correct'] += 1
                    snr_stats[snr]['loss'] += loss.item()
    
    # Calculate overall metrics
    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels)) * 100
    
    # Print SNR-specific results
    print(f"\n📊 {model_name} - Performance by SNR:")
    print("-" * 50)
    for snr in target_snrs:
        if snr_stats[snr]['total'] > 0:
            accuracy = snr_stats[snr]['correct'] / snr_stats[snr]['total'] * 100
            avg_loss = snr_stats[snr]['loss'] / snr_stats[snr]['total']
            print(f"SNR {snr:2d} dB: {accuracy:5.1f}% accuracy ({snr_stats[snr]['correct']:4d}/{snr_stats[snr]['total']:4d}) | Loss: {avg_loss:.4f}")
        else:
            print(f"SNR {snr:2d} dB: No data available")
    
    print(f"Overall: {overall_accuracy:5.1f}% accuracy")
    
    # Generate and save confusion matrices
    plot_confusion_matrices_by_snr(all_predictions, all_true_labels, all_snr_values, 
                                 target_snrs, target_modulations, model_name)
    
    return overall_accuracy, all_predictions, all_true_labels, all_snr_values

# ==============================================================================
# ## 6. Training Loop Utama
# ==============================================================================

# Initialize metrics and model states
metrics = {
    'standard': {'train_losses': [], 'train_accuracies': [], 'valid_losses': [], 'valid_accuracies': [], 'best_accuracy': 0.0, 'test_accuracy': 0.0},
    'parallel': {'train_losses': [], 'train_accuracies': [], 'valid_losses': [], 'valid_accuracies': [], 'best_accuracy': 0.0, 'test_accuracy': 0.0}
}
best_models = {'standard': None, 'parallel': None}
patience_counters = {'standard': 0, 'parallel': 0}

if model_standard and model_parallel and train_loader and valid_loader and test_loader:
    print("\n🎯 Memulai training komparatif...")
    print("=" * 80)

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        train_loss_std, train_acc_std = train_epoch(model_standard, train_loader, optimizer_standard, criterion, scaler_standard, device)
        train_loss_par, train_acc_par = train_epoch(model_parallel, train_loader, optimizer_parallel, criterion, scaler_parallel, device)

        # --- Validate ---
        valid_loss_std, valid_acc_std, _, _, _ = evaluate_epoch(model_standard, valid_loader, criterion, device)
        valid_loss_par, valid_acc_par, _, _, _ = evaluate_epoch(model_parallel, valid_loader, criterion, device)
        
        # --- Simpan Metrik ---
        metrics['standard']['train_losses'].append(train_loss_std)
        metrics['standard']['train_accuracies'].append(train_acc_std)
        metrics['standard']['valid_losses'].append(valid_loss_std)
        metrics['standard']['valid_accuracies'].append(valid_acc_std)
        
        metrics['parallel']['train_losses'].append(train_loss_par)
        metrics['parallel']['train_accuracies'].append(train_acc_par)
        metrics['parallel']['valid_losses'].append(valid_loss_par)
        metrics['parallel']['valid_accuracies'].append(valid_acc_par)
        
        # --- Update Scheduler ---
        scheduler_standard.step()
        scheduler_parallel.step()

        # --- Cek Model Terbaik & Early Stopping ---
        # Model Standard
        if valid_acc_std > metrics['standard']['best_accuracy']:
            metrics['standard']['best_accuracy'] = valid_acc_std
            best_models['standard'] = copy.deepcopy(model_standard.state_dict())
            patience_counters['standard'] = 0
        else:
            patience_counters['standard'] += 1
        
        # Model Parallel
        if valid_acc_par > metrics['parallel']['best_accuracy']:
            metrics['parallel']['best_accuracy'] = valid_acc_par
            best_models['parallel'] = copy.deepcopy(model_parallel.state_dict())
            patience_counters['parallel'] = 0
        else:
            patience_counters['parallel'] += 1

        # --- Tampilkan Hasil per Epoch ---
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Std: [Train: {train_acc_std:.2f}%, Valid: {valid_acc_std:.2f}%] | "
              f"Par: [Train: {train_acc_par:.2f}%, Valid: {valid_acc_par:.2f}%]")

        # --- Cek Early Stopping ---
        if patience_counters['standard'] >= patience and patience_counters['parallel'] >= patience:
            print(f"\n>> Early stopping di epoch {epoch+1} karena kedua model tidak menunjukkan peningkatan.")
            break

    print("\n🎉 Training Selesai!")
    print(f"Akurasi Validasi Terbaik (Standard): {metrics['standard']['best_accuracy']:.2f}%")
    print(f"Akurasi Validasi Terbaik (Parallel): {metrics['parallel']['best_accuracy']:.2f}%")

    # ==============================================================================
    # ## 7. Testing Final
    # ==============================================================================
    print("\n🔍 Melakukan pengujian final pada test set...")
    
    # Load model terbaik
    model_standard.load_state_dict(best_models['standard'])
    model_parallel.load_state_dict(best_models['parallel'])
    
    # Evaluasi kedua model dengan analisis detail per SNR
    print("\n" + "="*80)
    print("DETAILED TEST RESULTS")
    print("="*80)
    
    # Standard Model
    test_acc_std, pred_std, true_std, snr_std = evaluate_model_by_snr(
        model_standard, test_loader, criterion, device, 
        TARGET_SNR, TARGET_MODULATIONS, "Standard_CNN_LSTM"
    )
    
    # Parallel Model  
    test_acc_par, pred_par, true_par, snr_par = evaluate_model_by_snr(
        model_parallel, test_loader, criterion, device,
        TARGET_SNR, TARGET_MODULATIONS, "Parallel_CNN_LSTM"
    )
    
    # Simpan hasil tes
    metrics['standard']['test_accuracy'] = test_acc_std
    metrics['parallel']['test_accuracy'] = test_acc_par
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"📈 Test Accuracy - Standard Model: {test_acc_std:.2f}%")
    print(f"📈 Test Accuracy - Parallel Model: {test_acc_par:.2f}%")
    print(f"🏆 Best Model: {'Standard' if test_acc_std > test_acc_par else 'Parallel'}")
    
    # Generate overall classification reports
    print("\n" + "-"*50)
    print("CLASSIFICATION REPORT - STANDARD MODEL")
    print("-"*50)
    print(classification_report(true_std, pred_std, target_names=TARGET_MODULATIONS))
    
    print("\n" + "-"*50)  
    print("CLASSIFICATION REPORT - PARALLEL MODEL")
    print("-"*50)
    print(classification_report(true_par, pred_par, target_names=TARGET_MODULATIONS))
    
    # Create combined performance comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Accuracy by SNR comparison
    plt.subplot(2, 2, 1)
    snr_accs_std = []
    snr_accs_par = []
    
    for snr in TARGET_SNR:
        # Standard model accuracy for this SNR
        mask_std = np.array(snr_std) == snr
        if np.any(mask_std):
            acc_std = np.mean(np.array(pred_std)[mask_std] == np.array(true_std)[mask_std]) * 100
        else:
            acc_std = 0
        snr_accs_std.append(acc_std)
        
        # Parallel model accuracy for this SNR
        mask_par = np.array(snr_par) == snr
        if np.any(mask_par):
            acc_par = np.mean(np.array(pred_par)[mask_par] == np.array(true_par)[mask_par]) * 100
        else:
            acc_par = 0
        snr_accs_par.append(acc_par)
    
    x = np.arange(len(TARGET_SNR))
    width = 0.35
    
    plt.bar(x - width/2, snr_accs_std, width, label='Standard CNN-LSTM', alpha=0.8)
    plt.bar(x + width/2, snr_accs_par, width, label='Parallel CNN-LSTM', alpha=0.8)
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison by SNR')
    plt.xticks(x, [f'{snr}' for snr in TARGET_SNR])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training curves
    plt.subplot(2, 2, 2)
    epochs = range(1, len(metrics['standard']['valid_accuracies']) + 1)
    plt.plot(epochs, metrics['standard']['valid_accuracies'], 'b-', label='Standard - Validation', linewidth=2)
    plt.plot(epochs, metrics['parallel']['valid_accuracies'], 'r-', label='Parallel - Validation', linewidth=2)
    plt.plot(epochs, metrics['standard']['train_accuracies'], 'b--', label='Standard - Training', alpha=0.7)
    plt.plot(epochs, metrics['parallel']['train_accuracies'], 'r--', label='Parallel - Training', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss curves
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['standard']['valid_losses'], 'b-', label='Standard - Validation', linewidth=2)
    plt.plot(epochs, metrics['parallel']['valid_losses'], 'r-', label='Parallel - Validation', linewidth=2)
    plt.plot(epochs, metrics['standard']['train_losses'], 'b--', label='Standard - Training', alpha=0.7)
    plt.plot(epochs, metrics['parallel']['train_losses'], 'r--', label='Parallel - Training', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final metrics summary
    plt.subplot(2, 2, 4)
    models = ['Standard\nCNN-LSTM', 'Parallel\nCNN-LSTM']
    accuracies = [test_acc_std, test_acc_par]
    colors = ['skyblue', 'lightcoral']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Test Accuracy (%)')
    plt.title('Final Test Performance')
    plt.ylim(0, 100)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices/model_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Saved model comparison summary: confusion_matrices/model_comparison_summary.png")
    print(f"\n🎯 All confusion matrices and analysis saved in 'confusion_matrices/' directory")
    
else:
    print("\nTraining tidak dapat dimulai karena ada komponen yang tidak terdefinisi dengan benar.")
    if not (model_standard and model_parallel):
        print("- Model tidak berhasil dibuat")
    if not (train_loader and valid_loader and test_loader):
        print("- Data loader tidak berhasil dibuat")