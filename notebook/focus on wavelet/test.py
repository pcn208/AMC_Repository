import psutil
import os
import h5py

def check_file_conflicts():
    """Check if multiple processes are accessing HDF5 files"""
    
    DATA_DIR = 'focused_experiment_1d_iq_raw2'
    FEATURE_METHOD = 'wavelet_coeffs'
    
    target_files = [
        os.path.join(DATA_DIR, f'train_{FEATURE_METHOD}.h5'),
        os.path.join(DATA_DIR, f'valid_{FEATURE_METHOD}.h5')
    ]
    
    print("🔍 Checking for file access conflicts...")
    print("=" * 50)
    
    # Check running Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print(f"🐍 Found {len(python_processes)} Python processes:")
    for proc in python_processes:
        print(f"   PID {proc['pid']}: {proc['name']}")
        if proc['cmdline']:
            cmd = ' '.join(proc['cmdline'])
            if len(cmd) > 100:
                cmd = cmd[:100] + "..."
            print(f"      Command: {cmd}")
    
    # Check if files are locked
    print(f"\n📁 Checking file accessibility:")
    for file_path in target_files:
        if os.path.exists(file_path):
            try:
                # Try to open for reading
                with h5py.File(file_path, 'r') as f:
                    print(f"✅ {file_path} - OK (can read)")
                
                # Try to open for writing (this will fail if locked)
                try:
                    with h5py.File(file_path, 'r+') as f:
                        print(f"✅ {file_path} - OK (can write)")
                except:
                    print(f"⚠️ {file_path} - Locked for writing (probably in use)")
                    
            except Exception as e:
                print(f"❌ {file_path} - ERROR: {e}")
        else:
            print(f"📂 {file_path} - File doesn't exist")

def force_close_jupyter_handles():
    """Close any open HDF5 handles in Jupyter"""
    
    print("🧹 Attempting to close Jupyter file handles...")
    
    # Try to garbage collect
    import gc
    gc.collect()
    
    # Close any open HDF5 files
    try:
        # This works in some environments
        h5py.get_config().default_file_mode = 'r'
        print("✅ Reset HDF5 default mode")
    except:
        pass
    
    print("💡 If files are still locked:")
    print("   1. Restart your Jupyter kernel: Kernel → Restart")
    print("   2. Close Jupyter completely and reopen")
    print("   3. Run training script ONLY (not in Jupyter)")

def safe_file_cleanup():
    """Safely remove potentially corrupted files"""
    
    DATA_DIR = 'focused_experiment_1d_iq_raw2'
    FEATURE_METHOD = 'wavelet_coeffs'
    
    files_to_clean = [
        os.path.join(DATA_DIR, f'train_{FEATURE_METHOD}.h5'),
        os.path.join(DATA_DIR, f'valid_{FEATURE_METHOD}.h5')
    ]
    
    print("🗑️ Safe file cleanup...")
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                # Check if file is accessible
                with h5py.File(file_path, 'r') as f:
                    pass
                print(f"✅ {file_path} - File is OK, keeping it")
            except:
                print(f"🗑️ {file_path} - File corrupted, removing...")
                try:
                    os.remove(file_path)
                    print(f"   ✅ Removed successfully")
                except Exception as e:
                    print(f"   ❌ Could not remove: {e}")
                    print(f"   💡 Try closing all Python processes and manually delete")

if __name__ == "__main__":
    check_file_conflicts()
    print(f"\n" + "="*50)
    force_close_jupyter_handles()
    print(f"\n" + "="*50)
    safe_file_cleanup()