# Save as full_diagnostic.py and run in your new environment
import subprocess
import sys
import os

print("=" * 70)
print("FULL TENSORFLOW GPU DIAGNOSTIC")
print("=" * 70)

# 1. Environment info
print("\n1. PYTHON ENVIRONMENT:")
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda')}")

# 2. Check what TensorFlow is installed
print("\n2. TENSORFLOW INSTALLATION:")
result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'tensorflow'], 
                       capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'Name:' in line or 'Version:' in line:
        print(line)

result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'tensorflow-gpu'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("tensorflow-gpu is installed")
else:
    print("tensorflow-gpu is NOT installed")

# 3. Import TensorFlow and check build
print("\n3. TENSORFLOW BUILD INFO:")
try:
    import tensorflow as tf
    print(f"TF Version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Check if it's CPU or GPU build
    from tensorflow.python.platform import build_info as tf_build_info
    print(f"Build info: {tf_build_info.build_info}")
except Exception as e:
    print(f"Error: {e}")

# 4. Check CUDA files in conda env
print("\n4. CUDA FILES IN CONDA ENV:")
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if conda_prefix:
    cuda_paths = [
        os.path.join(conda_prefix, 'Library', 'bin'),
        os.path.join(conda_prefix, 'bin'),
        os.path.join(conda_prefix, 'lib'),
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            cuda_files = [f for f in files if 'cuda' in f.lower() or 'cudnn' in f.lower()]
            if cuda_files:
                print(f"\n{path}:")
                for f in cuda_files[:5]:  # Show first 5
                    print(f"  - {f}")

# 5. Check GPU visibility
print("\n5. GPU DETECTION ATTEMPTS:")
try:
    import tensorflow as tf
    
    # Method 1: Standard
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Method 1 - list_physical_devices: {gpus}")
    
    # Method 2: Experimental
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Method 2 - experimental.list_physical_devices: {gpus}")
    
    # Method 3: Low level
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    gpu_devices = [d for d in devices if d.device_type == 'GPU']
    print(f"Method 3 - device_lib: {gpu_devices}")
    
    # Method 4: Test availability
    print(f"Method 4 - is_gpu_available: {tf.test.is_gpu_available(cuda_only=True)}")
    
except Exception as e:
    print(f"Error during detection: {e}")

# 6. Check if CUDA DLLs are loadable
print("\n6. DLL LOADING TEST:")
import ctypes
dlls_to_check = [
    'cudart64_110.dll',
    'cudart64_11.dll', 
    'cublas64_11.dll',
    'cublasLt64_11.dll',
    'cudnn64_8.dll',
    'cufft64_10.dll'
]

for dll in dlls_to_check:
    try:
        ctypes.CDLL(dll)
        print(f"✓ {dll} loaded successfully")
    except:
        print(f"✗ {dll} failed to load")