import h5py
import numpy as np
import os

def check_dataset_contents():
    """Check what's actually in your HDF5 files"""
    
    DATA_DIR = 'focused_experiment_1d_iq_raw2'
    FEATURE_METHOD = 'wavelet_coeffs'
    
    train_path = os.path.join(DATA_DIR, f'train_{FEATURE_METHOD}.h5')
    valid_path = os.path.join(DATA_DIR, f'valid_{FEATURE_METHOD}.h5')
    
    for file_path in [train_path, valid_path]:
        if os.path.exists(file_path):
            print(f"\n📁 Checking: {file_path}")
            print("=" * 50)
            
            with h5py.File(file_path, 'r') as f:
                # Print all keys in the file
                print("🔑 Keys in file:")
                for key in f.keys():
                    print(f"   {key}: {f[key].shape}")
                
                # Check labels
                if 'labels' in f:
                    labels = f['labels'][:]
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    print(f"\n🏷️ Label distribution:")
                    print(f"   Unique labels: {unique_labels}")
                    print(f"   Counts: {counts}")
                    print(f"   Total classes: {len(unique_labels)}")
                
                # Check attributes
                print(f"\n📋 Attributes:")
                for attr_name in f.attrs:
                    attr_value = f.attrs[attr_name]
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8')
                    print(f"   {attr_name}: {attr_value}")
                
                # Check SNRs if available
                if 'snrs' in f:
                    snrs = f['snrs'][:]
                    unique_snrs = np.unique(snrs)
                    print(f"\n📶 SNR values: {unique_snrs}")
        else:
            print(f"❌ File not found: {file_path}")

# Run the check
check_dataset_contents()