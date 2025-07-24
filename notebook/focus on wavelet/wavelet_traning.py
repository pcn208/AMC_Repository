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
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm, trange
import seaborn as sns
import gc
import time
import pywt
import math
from collections import deque, defaultdict
import psutil
import os
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# PyTorch 2.6+ Fix: Added weights_only=False to torch.load calls
# This is safe for your own trained models but needed due to PyTorch security changes

# Focused experiment setup - FIXED TO MATCH ACTUAL DATA
TARGET_MODULATIONS = ['8PSK', '16QAM','BPSK']#ly 2 modulations actually in dataset (despite metadata claiming 4)
TARGET_SNRS = [-6, 6, 18]  # Low, Medium, High 

# Training configuration
BATCH_SIZE = 512
LEARNING_RATE = 0.0015
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience
MIN_DELTA = 0.001  # Minimum improvement for early stopping

# Model configuration
DROPOUT_RATE = 0.3
NUM_CLASSES = len(TARGET_MODULATIONS)  # Now correctly 2 classes
FEATURE_METHOD = 'wavelet_coeffs'

# Data paths
DATA_DIR = 'focused_experiment_1d_iq_raw2'  # Updated to match your actual directory

# Output paths
RESULTS_DIR = './training_results'
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, f'best_model_{FEATURE_METHOD}.pth')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class CNN1D_Shallow(nn.Module):
    """Enhanced shallow branch with more layers"""
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch, 1, seq_len)
        x = self.conv_block1(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        return x  # Keep temporal dimension

class CNN1D_Deep(nn.Module):
    """Enhanced deep branch with more layers"""
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
        )
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv_block3(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        return x  # Keep temporal dimension

class DiagramModel1D(nn.Module):
    """1D model with proper LSTM integration"""
    def __init__(self, input_size, num_classes, dropout_rate=0.4):
        super().__init__()
        self.cnn1 = CNN1D_Shallow(input_size, dropout_rate)
        self.cnn2 = CNN1D_Deep(input_size, dropout_rate)
        
        # Calculate output channels after CNN processing
        self.shallow_out_channels = 128
        self.deep_out_channels = 256
        
        # LSTM input size (shallow + deep channels)
        lstm_input_size = self.shallow_out_channels + self.deep_out_channels
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0.0  # Fix for single layer
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),  # 256*2 (bidirectional)
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Process through both branches
        shallow_out = self.cnn1(x)  # (batch, 128, seq_len/4)
        deep_out = self.cnn2(x)     # (batch, 256, seq_len/8)
        
        # Adjust sequence lengths using adaptive pooling
        target_length = min(shallow_out.size(2), deep_out.size(2))
        if target_length <= 0:
            target_length = 1
            
        shallow_out = F.adaptive_avg_pool1d(shallow_out, target_length)
        deep_out = F.adaptive_avg_pool1d(deep_out, target_length)
        
        # Concatenate along channel dimension
        fused = torch.cat((shallow_out, deep_out), dim=1)  # (batch, 384, seq_len)
        
        # Permute for LSTM: (batch, seq_len, features)
        lstm_input = fused.permute(0, 2, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)  # (batch, seq_len, 512)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, 512)
        
        # Classification
        output = self.classifier(context_vector)
        return output

class DiagramIQModel1D(nn.Module):
    """1D model for I/Q dual-channel processing"""
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super().__init__()
        self.feature1_branch = DiagramModel1D(input_size, num_classes, dropout_rate)
        self.feature2_branch = DiagramModel1D(input_size, num_classes, dropout_rate)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, feature1, feature2):
        out1 = self.feature1_branch(feature1)
        out2 = self.feature2_branch(feature2)
        
        # Concatenate and fuse
        combined = torch.cat([out1, out2], dim=1)
        output = self.fusion(combined)
        return output
    
class RadioMLDataset(Dataset):
    """Dataset class for RadioML 1D preprocessed data"""
    
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        
        # Check if file exists
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        # Check file size (should not be empty)
        file_size = os.path.getsize(hdf5_path)
        if file_size == 0:
            raise ValueError(f"HDF5 file is empty: {hdf5_path}")
        
        print(f"📁 Loading HDF5 file: {hdf5_path} ({file_size/1024/1024:.1f} MB)")
        
        try:
            # Try to open and validate the HDF5 file
            with h5py.File(hdf5_path, 'r') as f:
                # Check if required datasets exist
                required_keys = ['feature1', 'feature2', 'labels', 'snrs']
                missing_keys = [key for key in required_keys if key not in f.keys()]
                
                if missing_keys:
                    raise KeyError(f"Missing required datasets in HDF5 file: {missing_keys}")
                
                # Load data arrays
                print("   Loading datasets...")
                self.feature1 = f['feature1'][:]
                self.feature2 = f['feature2'][:]
                self.labels = f['labels'][:]
                self.snrs = f['snrs'][:]
                
                # Validate data shapes
                if len(set([len(self.feature1), len(self.feature2), len(self.labels), len(self.snrs)])) != 1:
                    raise ValueError("Inconsistent dataset lengths in HDF5 file")
                
                print("   Loading metadata...")
                # Load metadata with error handling
                try:
                    self.num_samples = f.attrs.get('num_samples', len(self.labels))
                    self.feature_method = f.attrs.get('feature_method', 'unknown')
                    if isinstance(self.feature_method, bytes):
                        self.feature_method = self.feature_method.decode('utf-8')
                        
                    self.feature_length = f.attrs.get('feature_length', self.feature1.shape[1])
                except Exception as attr_error:
                    print(f"   ⚠️ Warning: Could not load some metadata: {attr_error}")
                    # Use defaults if metadata is corrupted
                    self.num_samples = len(self.labels)
                    self.feature_method = 'unknown'
                    self.feature_length = self.feature1.shape[1]
                    
        except Exception as e:
            print(f"❌ Error loading HDF5 file {hdf5_path}:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            
            # Provide helpful suggestions
            print(f"\n🔧 Troubleshooting suggestions:")
            print(f"   1. File might be corrupted - try regenerating it")
            print(f"   2. File might be locked by another process")
            print(f"   3. Check if preprocessing completed successfully")
            print(f"   4. Try deleting the file and running preprocessing again")
            
            raise RuntimeError(f"Failed to load HDF5 file: {hdf5_path}. See suggestions above.") from e
        
        print(f"📊 Loaded dataset: {self.num_samples} samples")
        print(f"   Feature method: {self.feature_method}")
        print(f"   Feature shapes: {self.feature1.shape}, {self.feature2.shape}")
        
        # Class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        self.actual_num_classes = len(unique)  # Store actual number of classes
        
        for label, count in zip(unique, counts):
            if label < len(TARGET_MODULATIONS):
                mod_name = TARGET_MODULATIONS[label]
            else:
                mod_name = f"Class_{label}"
            print(f"   {mod_name}: {count} samples")
        
        # Warn if metadata doesn't match actual data (only if metadata was loaded)
        try:
            with h5py.File(hdf5_path, 'r') as f:
                metadata_classes = f.attrs.get('num_classes', len(unique))
                if metadata_classes != self.actual_num_classes:
                    print(f"⚠️  WARNING: Metadata claims {metadata_classes} classes, but data has {self.actual_num_classes} classes!")
        except:
            print("⚠️  WARNING: Could not verify metadata consistency")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature1 = torch.FloatTensor(self.feature1[idx])
        feature2 = torch.FloatTensor(self.feature2[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        snr = self.snrs[idx]
        
        return feature1, feature2, label, snr

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.counter >= self.patience

def create_data_loaders():
    """Create train and validation data loaders"""
    train_path = os.path.join(DATA_DIR, f'train_{FEATURE_METHOD}.h5')
    valid_path = os.path.join(DATA_DIR, f'valid_{FEATURE_METHOD}.h5')
    
    # Check file existence
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Validation data not found: {valid_path}")
    
    # Create datasets and loaders
    train_dataset = RadioMLDataset(train_path)
    valid_dataset = RadioMLDataset(valid_path)
    
    # Use actual number of classes from the data
    actual_num_classes = train_dataset.actual_num_classes
    print(f"🎯 Using {actual_num_classes} classes for training")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,  # Reduced for better stability
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,  # Reduced for better stability
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    return train_loader, valid_loader, train_dataset.feature_length, actual_num_classes

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for feature1, feature2, labels, snrs in pbar:
        try:
            feature1 = feature1.to(device)
            feature2 = feature2.to(device) 
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feature1, feature2)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        except Exception as e:
            print(f"Error in training batch: {str(e)}")
            continue
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy

def validate_epoch(model, valid_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validation")
        for feature1, feature2, labels, snrs in pbar:
            try:
                feature1 = feature1.to(device)
                feature2 = feature2.to(device)
                labels = labels.to(device)
                
                outputs = model(feature1, feature2)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_snrs.extend(snrs.numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue
    
    avg_loss = total_loss / len(valid_loader) if len(valid_loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy, all_predictions, all_labels, all_snrs

def train_model():
    """Main training function"""
    print(f"🎯 Starting Training: {', '.join(TARGET_MODULATIONS)}")
    print("=" * 60)
    
    try:
        # Create data loaders
        train_loader, valid_loader, feature_length, actual_num_classes = create_data_loaders()
        
        # Initialize model with actual number of classes
        model = DiagramIQModel1D(
            input_size=feature_length,
            num_classes=actual_num_classes,  # Use actual classes from data
            dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 Model: DiagramIQModel1D")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Feature length: {feature_length}")
        print(f"   Classes: {actual_num_classes} (actual data) vs {NUM_CLASSES} (config)")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
        
        # Training history
        history = defaultdict(list)
        best_val_acc = 0
        best_epoch = 0
        
        print(f"\n🚀 Training started...")
        start_time = time.time()
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n📈 Epoch {epoch+1}/{NUM_EPOCHS}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            
            # Validation  
            val_loss, val_acc, val_preds, val_labels, val_snrs = validate_epoch(
                model, valid_loader, criterion, DEVICE
            )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'feature_method': FEATURE_METHOD,
                    'num_classes': actual_num_classes,  # Save actual classes
                    'feature_length': feature_length
                }, MODEL_SAVE_PATH)
                print(f"💾 New best model saved! (Acc: {val_acc:.2f}%)")
            
            # Early stopping check
            if early_stopping(val_acc):
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        
        return history, model
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

def evaluate_model():
    """Evaluate the trained model and create visualizations"""
    print("\n🔍 Evaluating trained model...")
    
    try:
        # Load data to get dimensions
        train_loader, valid_loader, feature_length, actual_num_classes = create_data_loaders()
        
        # Load checkpoint to get saved model configuration
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_SAVE_PATH}")
            
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
        saved_num_classes = checkpoint.get('num_classes', actual_num_classes)
        
        # Load best model
        model = DiagramIQModel1D(
            input_size=feature_length,
            num_classes=saved_num_classes,  # Use saved num_classes
            dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📁 Loaded model from epoch {checkpoint['epoch']+1}")
        
        # Evaluate
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, val_preds, val_labels, val_snrs = validate_epoch(
            model, valid_loader, criterion, DEVICE
        )
        
        # Calculate metrics
        if len(val_labels) > 0 and len(val_preds) > 0:
            f1 = f1_score(val_labels, val_preds, average='weighted')
            
            print(f"📊 Final Results:")
            print(f"   Accuracy: {val_acc:.2f}%")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Loss: {val_loss:.4f}")
            
            # Classification report
            class_names = TARGET_MODULATIONS[:saved_num_classes]  # Use saved_num_classes
            print(f"\n📋 Classification Report:")
            print(classification_report(val_labels, val_preds, target_names=class_names))
            
            # Confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
            plt.show()
            
            # Performance by SNR
            snr_results = {}
            for snr in np.unique(val_snrs):
                mask = np.array(val_snrs) == snr
                if np.sum(mask) > 0:
                    snr_acc = accuracy_score(np.array(val_labels)[mask], np.array(val_preds)[mask])
                    snr_results[snr] = snr_acc * 100
                    print(f"   SNR {snr:2.0f}dB: {snr_acc*100:.2f}%")
            
            # Plot SNR performance
            if snr_results:
                plt.figure(figsize=(10, 6))
                snrs = sorted(snr_results.keys())
                accs = [snr_results[snr] for snr in snrs]
                
                plt.plot(snrs, accs, 'bo-', linewidth=2, markersize=8)
                plt.xlabel('SNR (dB)')
                plt.ylabel('Accuracy (%)')
                plt.title('Classification Accuracy vs SNR')
                plt.grid(True, alpha=0.3)
                plt.ylim([0, 100])
                
                for snr, acc in zip(snrs, accs):
                    plt.annotate(f'{acc:.1f}%', (snr, acc), textcoords="offset points", 
                                xytext=(0,10), ha='center')
                
                plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_vs_snr.png'), dpi=150, bbox_inches='tight')
                plt.show()
            
            return val_acc, f1
        else:
            print("❌ No validation data available for evaluation")
            return 0, 0
            
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise

def plot_training_history(history):
    """Plot training history"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(history['lr'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Best accuracy highlight
        if history['val_acc']:
            best_val_idx = np.argmax(history['val_acc'])
            best_val_acc = history['val_acc'][best_val_idx]
            
            axes[1, 1].plot(history['val_acc'], color='red', linewidth=2)
            axes[1, 1].scatter(best_val_idx, best_val_acc, color='gold', s=100, zorder=5)
            axes[1, 1].set_title(f'Best Validation Accuracy: {best_val_acc:.2f}%')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation Accuracy (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"❌ Error plotting training history: {str(e)}")

def main():
    """Main function to run the complete training pipeline"""
    try:
        print(f"🎯 RadioML Training Pipeline")
        print(f"Target Modulations: {TARGET_MODULATIONS}")
        print(f"Feature Method: {FEATURE_METHOD}")
        print(f"Device: {DEVICE}")
        print("=" * 60)
        
        # Train the model
        history, model = train_model()
        
        # Save training history
        history_file = os.path.join(RESULTS_DIR, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"💾 Training history saved to: {history_file}")
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate the model
        final_acc, final_f1 = evaluate_model()
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   Final Accuracy: {final_acc:.2f}%")
        print(f"   Final F1-Score: {final_f1:.4f}")
        
        return history, model, final_acc, final_f1
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()