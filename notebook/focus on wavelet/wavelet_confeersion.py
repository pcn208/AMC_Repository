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
from collections import deque
import psutil
import os
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "C:\\workarea\\CNN model\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
JSON_PATH = 'C:\\workarea\\CNN model\\dataset\\radioml2018\\versions\\2\\classes-fixed.json' 

TARGET_MODULATIONS = ['8PSK', '16QAM', 'BPSK']  # All 4 modulations
TARGET_SNRS = [-6, 6, 18]  # Low, Medium, High 

# Processing options
FEATURE_METHOD = 'wavelet_coeffs'  # Options: 'amplitude_phase', 'iq_raw', 'fft_features', 'wavelet_coeffs', 'statistical'
OUTPUT_DIR = './focused_experiment_1d_iq_raw2'
TRAIN_RATIO = 0.7
VALID_RATIO = 0.3

# Feature extraction settings
TARGET_LENGTH = 1024  # Standard 1D feature length
NUM_FFT_BINS = 256   # Number of FFT frequency bins to keep
NUM_WAVELET_LEVELS = 6  # Wavelet decomposition levels

print(f"🎯 Focused 1D Experiment Setup:")
print(f"   Modulations: {TARGET_MODULATIONS}")
print(f"   SNR levels: {TARGET_SNRS}")
print(f"   Feature method: {FEATURE_METHOD}")
print(f"   Target length: {TARGET_LENGTH}")
print(f"   Output directory: {OUTPUT_DIR}")

def load_and_filter_data():
    """Load RadioML data and filter for target modulations and SNRs."""
    
    print("📂 Loading and filtering RadioML data...")
    
    # Load files
    h5_file = h5py.File(FILE_PATH, 'r')
    with open(JSON_PATH, 'r') as f:
        modulation_classes = json.load(f)
    
    # Load arrays
    X = h5_file['X']  # Shape: (samples, 1024, 2)
    Y = np.argmax(h5_file['Y'], axis=1)  # Labels
    Z = h5_file['Z'][:, 0]  # SNRs
    
    print(f"   Original dataset: {X.shape[0]} samples")
    
    # Get target modulation indices
    target_mod_indices = [modulation_classes.index(mod) for mod in TARGET_MODULATIONS]
    print(f"   Target modulation indices: {dict(zip(TARGET_MODULATIONS, target_mod_indices))}")
    
    # Filter for target modulations and SNRs
    mask = np.zeros(len(Y), dtype=bool)
    
    for mod_idx in target_mod_indices:
        for snr in TARGET_SNRS:
            condition = (Y == mod_idx) & (Z == snr)
            mask |= condition
            count = np.sum(condition)
            mod_name = TARGET_MODULATIONS[target_mod_indices.index(mod_idx)]
            print(f"   {mod_name} at {snr}dB: {count} samples")
    
    # Apply filter
    X_filtered = X[mask]
    Y_filtered = Y[mask].copy()  # Make a copy to avoid issues
    Z_filtered = Z[mask]
    
    # FIXED: Proper label remapping to sequential indices (0, 1, 2, 3)
    label_mapping = {}
    Y_remapped = np.zeros_like(Y_filtered)
    
    for new_label, old_label in enumerate(target_mod_indices):
        old_mask = Y_filtered == old_label
        Y_remapped[old_mask] = new_label
        label_mapping[old_label] = new_label
        count = np.sum(old_mask)
        print(f"   Remapped {TARGET_MODULATIONS[new_label]} ({old_label} -> {new_label}): {count} samples")
    
    # Verify remapping worked
    unique_new_labels = np.unique(Y_remapped)
    print(f"   Final label distribution: {unique_new_labels}")
    
    print(f"✅ Filtered dataset: {X_filtered.shape[0]} samples")
    print(f"   Label mapping: {label_mapping}")
    
    h5_file.close()
    return X_filtered, Y_remapped, Z_filtered, label_mapping

def extract_amplitude_phase_features(i_signal, q_signal):
    """Extract amplitude and phase as 1D time series."""
    
    # Calculate amplitude and phase
    amplitude = np.sqrt(i_signal**2 + q_signal**2)
    phase = np.arctan2(q_signal, i_signal)
    
    # Unwrap phase to avoid discontinuities
    phase_unwrapped = np.unwrap(phase)
    
    # Resize to target length
    amplitude_resized = resize_to_target_length(amplitude)
    phase_resized = resize_to_target_length(phase_unwrapped)
    
    return amplitude_resized.astype(np.float32), phase_resized.astype(np.float32)

def extract_iq_raw_features(i_signal, q_signal):
    """Simple I/Q channel features (minimal processing)."""
    
    # Just resize the raw I/Q signals
    i_resized = resize_to_target_length(i_signal)
    q_resized = resize_to_target_length(q_signal)
    
    return i_resized.astype(np.float32), q_resized.astype(np.float32)

def extract_fft_features(i_signal, q_signal):
    """Extract FFT magnitude and phase as 1D features."""
    
    # Combine I/Q into complex signal
    complex_signal = i_signal + 1j * q_signal
    
    # Compute FFT
    fft_result = fft(complex_signal)
    
    # Extract magnitude and phase
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # Take only the first half (positive frequencies)
    half_length = len(magnitude) // 2
    magnitude = magnitude[:half_length]
    phase = phase[:half_length]
    
    # Resize to target length
    magnitude_resized = resize_to_target_length(magnitude)
    phase_resized = resize_to_target_length(phase)
    
    return magnitude_resized.astype(np.float32), phase_resized.astype(np.float32)

def extract_wavelet_coeffs(i_signal, q_signal):
    """Extract 1D wavelet coefficients."""
    
    # Apply DWT to I and Q channels
    wavelet = 'db4'
    
    # Multi-level wavelet decomposition
    coeffs_i = pywt.wavedec(i_signal, wavelet, level=NUM_WAVELET_LEVELS)
    coeffs_q = pywt.wavedec(q_signal, wavelet, level=NUM_WAVELET_LEVELS)
    
    # Concatenate all coefficients into 1D arrays
    i_features = np.concatenate(coeffs_i)
    q_features = np.concatenate(coeffs_q)
    
    # Resize to target length
    i_resized = resize_to_target_length(i_features)
    q_resized = resize_to_target_length(q_features)
    
    return i_resized.astype(np.float32), q_resized.astype(np.float32)

def extract_statistical_features(i_signal, q_signal):
    """Extract statistical features using sliding windows."""
    
    # Window size for statistical calculations
    window_size = 64
    hop_size = 32
    
    # Calculate sliding window statistics
    i_stats = []
    q_stats = []
    
    for start in range(0, len(i_signal) - window_size + 1, hop_size):
        end = start + window_size
        
        i_window = i_signal[start:end]
        q_window = q_signal[start:end]
        
        # Statistical features for each window
        i_stats.extend([
            np.mean(i_window), np.std(i_window), 
            np.var(i_window), np.max(i_window), np.min(i_window),
            np.median(i_window), np.ptp(i_window)  # peak-to-peak
        ])
        
        q_stats.extend([
            np.mean(q_window), np.std(q_window),
            np.var(q_window), np.max(q_window), np.min(q_window),
            np.median(q_window), np.ptp(q_window)
        ])
    
    i_stats = np.array(i_stats)
    q_stats = np.array(q_stats)
    
    # Resize to target length
    i_resized = resize_to_target_length(i_stats)
    q_resized = resize_to_target_length(q_stats)
    
    return i_resized.astype(np.float32), q_resized.astype(np.float32)

def safe_decode_attr(attr):
    """Safely decode HDF5 attribute (handles both string and bytes)."""
    if isinstance(attr, bytes):
        return attr.decode('utf-8')
    return str(attr)

def resize_to_target_length(signal_1d):
    """Resize 1D signal to target length using interpolation."""
    
    current_length = len(signal_1d)
    
    if current_length == TARGET_LENGTH:
        return signal_1d
    
    # Create interpolation indices
    old_indices = np.linspace(0, current_length - 1, current_length)
    new_indices = np.linspace(0, current_length - 1, TARGET_LENGTH)
    
    # Interpolate
    resized = np.interp(new_indices, old_indices, signal_1d)
    
    return resized

def preprocess_focused_dataset():
    """Main preprocessing pipeline for focused 1D experiment."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and filter data
    X_filtered, Y_filtered, Z_filtered, label_mapping = load_and_filter_data()
    
    total_samples = len(X_filtered)
    print(f"\n🌊 Processing {total_samples} samples with {FEATURE_METHOD} method...")
    
    # Select feature extraction method
    if FEATURE_METHOD == 'amplitude_phase':
        extract_features = extract_amplitude_phase_features
        feature_names = ['amplitude', 'phase']
    elif FEATURE_METHOD == 'iq_raw':
        extract_features = extract_iq_raw_features
        feature_names = ['i_channel', 'q_channel']
    elif FEATURE_METHOD == 'fft_features':
        extract_features = extract_fft_features
        feature_names = ['fft_magnitude', 'fft_phase']
    elif FEATURE_METHOD == 'wavelet_coeffs':
        extract_features = extract_wavelet_coeffs
        feature_names = ['i_wavelets', 'q_wavelets']
    elif FEATURE_METHOD == 'statistical':
        extract_features = extract_statistical_features
        feature_names = ['i_statistics', 'q_statistics']
    else:
        raise ValueError(f"Unknown feature method: {FEATURE_METHOD}")
    
    # Process all samples
    start_time = time.time()
    
    feature1_list = []
    feature2_list = []
    labels_list = []
    snrs_list = []
    
    for i in tqdm(range(total_samples), desc="Processing samples"):
        # Extract I/Q signals
        i_signal = X_filtered[i, :, 0]
        q_signal = X_filtered[i, :, 1]
        
        # Apply feature extraction
        feature1, feature2 = extract_features(i_signal, q_signal)
        
        # Store results
        feature1_list.append(feature1)
        feature2_list.append(feature2)
        labels_list.append(Y_filtered[i])
        snrs_list.append(Z_filtered[i])
    
    # Convert to numpy arrays
    features1 = np.array(feature1_list)
    features2 = np.array(feature2_list)
    labels = np.array(labels_list)
    snrs = np.array(snrs_list)
    
    processing_time = time.time() - start_time
    print(f"✅ Processing complete in {processing_time/60:.1f} minutes")
    print(f"   Features shape: {features1.shape}, {features2.shape}")
    print(f"   Speed: {total_samples/processing_time:.1f} samples/second")
    
    # Verify all classes are present before splitting
    unique_labels = np.unique(labels)
    print(f"\n🔍 Pre-split verification:")
    print(f"   Unique labels in processed data: {unique_labels}")
    print(f"   Expected labels: {list(range(len(TARGET_MODULATIONS)))}")
    
    if len(unique_labels) != len(TARGET_MODULATIONS):
        print(f"⚠️ WARNING: Expected {len(TARGET_MODULATIONS)} classes, found {len(unique_labels)}")
        print("   This might indicate missing data for some modulations/SNRs")
    
    # Create train/valid splits
    create_train_valid_splits(features1, features2, labels, snrs, feature_names)
    
    return features1, features2, labels, snrs

def create_train_valid_splits(features1, features2, labels, snrs, feature_names):
    """Create train/validation splits and save to HDF5."""
    
    print(f"\n📊 Creating train/validation splits...")
    
    # Stratified split to maintain modulation/SNR balance
    train_indices = []
    valid_indices = []
    
    # Get all unique labels in the data (dynamic, not hardcoded!)
    unique_labels = np.unique(labels)
    print(f"   Processing {len(unique_labels)} modulation classes: {unique_labels}")
    
    # FIXED: Split by all modulations present in data, not just [0, 1]
    for mod_label in unique_labels:
        for snr in TARGET_SNRS:
            # Find samples for this combination
            mask = (labels == mod_label) & (snrs == snr)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:  # Only process if data exists
                # Shuffle and split
                np.random.seed(42)  # For reproducibility
                np.random.shuffle(indices)
                split_point = int(TRAIN_RATIO * len(indices))
                
                train_indices.extend(indices[:split_point])
                valid_indices.extend(indices[split_point:])
                
                mod_name = TARGET_MODULATIONS[mod_label] if mod_label < len(TARGET_MODULATIONS) else f"Class_{mod_label}"
                print(f"   {mod_name} at {snr}dB: {len(indices)} total, {split_point} train, {len(indices)-split_point} valid")
            else:
                mod_name = TARGET_MODULATIONS[mod_label] if mod_label < len(TARGET_MODULATIONS) else f"Class_{mod_label}"
                print(f"   ⚠️ No data found for {mod_name} at {snr}dB")
    
    train_indices = np.array(train_indices)
    valid_indices = np.array(valid_indices)
    
    print(f"\n   📊 Final split summary:")
    print(f"      Train samples: {len(train_indices)}")
    print(f"      Valid samples: {len(valid_indices)}")
    
    # Verify class distribution in splits
    train_labels = labels[train_indices]
    valid_labels = labels[valid_indices]
    
    print(f"   🧮 Train class distribution:")
    for label in unique_labels:
        count = np.sum(train_labels == label)
        mod_name = TARGET_MODULATIONS[label] if label < len(TARGET_MODULATIONS) else f"Class_{label}"
        print(f"      {mod_name}: {count} samples")
    
    print(f"   🧮 Valid class distribution:")
    for label in unique_labels:
        count = np.sum(valid_labels == label)
        mod_name = TARGET_MODULATIONS[label] if label < len(TARGET_MODULATIONS) else f"Class_{label}"
        print(f"      {mod_name}: {count} samples")
    
    # Save train split
    save_split('train', train_indices, features1, features2, labels, snrs, feature_names)
    
    # Save validation split
    save_split('valid', valid_indices, features1, features2, labels, snrs, feature_names)

def save_split(split_name, indices, features1, features2, labels, snrs, feature_names):
    """Save a data split to HDF5 file."""
    
    output_file = os.path.join(OUTPUT_DIR, f'{split_name}_{FEATURE_METHOD}.h5')
    
    with h5py.File(output_file, 'w') as f:
        # Save features
        f.create_dataset('feature1', data=features1[indices], compression='gzip')
        f.create_dataset('feature2', data=features2[indices], compression='gzip')
        f.create_dataset('labels', data=labels[indices])
        f.create_dataset('snrs', data=snrs[indices])
        
        # Save metadata
        f.attrs['num_samples'] = len(indices)
        f.attrs['feature_method'] = FEATURE_METHOD
        f.attrs['feature1_name'] = feature_names[0]
        f.attrs['feature2_name'] = feature_names[1]
        f.attrs['modulations'] = TARGET_MODULATIONS
        f.attrs['target_snrs'] = TARGET_SNRS
        f.attrs['num_classes'] = len(TARGET_MODULATIONS)
        f.attrs['feature_length'] = TARGET_LENGTH
        f.attrs['is_1d'] = True
        
        # ADDED: Save actual class distribution for verification
        unique_labels_in_split = np.unique(labels[indices])
        f.attrs['actual_classes_in_split'] = unique_labels_in_split
        f.attrs['actual_num_classes_in_split'] = len(unique_labels_in_split)
    
    file_size = os.path.getsize(output_file) / (1024**2)  # MB
    print(f"   ✅ {split_name}: {output_file} ({file_size:.1f} MB)")

def visualize_1d_samples():
    """Visualize sample 1D features to verify processing."""
    
    print(f"\n🔍 Visualizing sample 1D features...")
    
    # Load train data
    train_file = os.path.join(OUTPUT_DIR, f'train_{FEATURE_METHOD}.h5')
    
    with h5py.File(train_file, 'r') as f:
        # Get samples from each class for visualization
        all_labels = f['labels'][:]
        unique_labels = np.unique(all_labels)
        
        sample_indices = []
        for label in unique_labels[:6]:  # Max 6 samples for visualization
            class_indices = np.where(all_labels == label)[0]
            if len(class_indices) > 0:
                sample_indices.append(class_indices[0])  # Take first sample of each class
        
        if len(sample_indices) < 6:
            # Fill remaining spots with random samples
            remaining_needed = 6 - len(sample_indices)
            additional_indices = np.random.choice(len(all_labels), remaining_needed, replace=False)
            sample_indices.extend(additional_indices)
        
        sample_indices = sample_indices[:6]  # Ensure exactly 6 samples
        
        features1 = f['feature1'][sample_indices]
        features2 = f['feature2'][sample_indices]
        labels = f['labels'][sample_indices]
        snrs = f['snrs'][sample_indices]
        
        # Handle string attributes safely
        feature1_name = safe_decode_attr(f.attrs['feature1_name'])
        feature2_name = safe_decode_attr(f.attrs['feature2_name'])
    
    # Plot samples
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Sample 1D Features: {FEATURE_METHOD}', fontsize=16)
    
    for i in range(6):
        row = i // 2
        col = i % 2
        
        # Plot both features for each sample
        x_axis = np.arange(len(features1[i]))
        
        axes[row, col].plot(x_axis, features1[i], label=feature1_name, alpha=0.8, linewidth=1)
        axes[row, col].plot(x_axis, features2[i], label=feature2_name, alpha=0.8, linewidth=1)
        
        label_idx = labels[i]
        mod_name = TARGET_MODULATIONS[label_idx] if label_idx < len(TARGET_MODULATIONS) else f"Class_{label_idx}"
        axes[row, col].set_title(f'{mod_name} at {snrs[i]:.0f}dB')
        axes[row, col].set_xlabel('Sample Index')
        axes[row, col].set_ylabel('Amplitude')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'sample_1d_features_{FEATURE_METHOD}.png'), dpi=150)
    plt.show()
    
    print(f"✅ Sample 1D visualization saved")

if __name__ == "__main__":
    # Run preprocessing
    preprocess_focused_dataset()
    # Visualize sample features
    visualize_1d_samples()
    print("\n🎉 Focused 1D experiment preprocessing complete!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Processed features saved for {FEATURE_METHOD} method.")