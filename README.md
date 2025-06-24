# CNN Models for I/Q Signal Modulation Classification

## Project Overview

This project implements Convolutional Neural Networks (CNNs) for automatic modulation classification (AMC) using I/Q signal data. The goal is to classify different digital modulation schemes from raw I/Q samples.

## Dataset Information

- **Dataset**: RadioML 2018.01A
- **Signal Types**: 8 modulation classes
  - OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK
- **Signal Format**: I/Q samples (In-phase and Quadrature components)
- **Frame Size**: 1024 samples per signal
- **SNR Range**: -30dB to +30dB (26 SNR levels)
- **Total Samples**: ~2.5M samples across all modulations and SNR levels

### Data Channel Structure
```
Input Data Shape: (batch_size, 2, 1024)
├── Channel 0: I (In-phase) component
└── Channel 1: Q (Quadrature) component
```

The **2 input channels** represent the complex baseband signal:
- **I Channel**: Real part of the complex signal (cosine component)
- **Q Channel**: Imaginary part of the complex signal (sine component)

## Model Architectures Developed

### 1. CNN-32 Channel Model (`NewCNN32`)

**Architecture Philosophy**: Conservative approach with maximum 32 feature channels to prevent overfitting.

**Channel Progression**: 2 → 16 → 32 → 32 → 24 → 16 → 8

```python
Model Architecture:
Input: (batch, 2, 1024)           # I/Q signal data
├── Conv1d: 2 → 16 channels       # Initial feature extraction
├── Conv1d: 16 → 32 channels      # Max channel capacity
├── Conv1d: 32 → 32 channels      # Maintain feature richness
├── Conv1d: 32 → 24 channels      # Begin channel reduction
├── Conv1d: 24 → 16 channels      # Further reduction
├── Conv1d: 16 → 8 channels       # Final feature consolidation
└── Classifier: 8*4 → 128 → 64 → 8 classes
```

**Model Statistics**:
- **Parameters**: ~150K
- **Memory Usage**: ~3.5GB (batch size 256)
- **Target Use**: Baseline model, overfitting prevention

### 2. CNN-64 Channel Model (`NewCNN64`)

**Architecture Philosophy**: Higher capacity model with maximum 64 feature channels for improved pattern recognition.

**Channel Progression**: 2 → 32 → 64 → 64 → 48 → 32 → 16

```python
Model Architecture:
Input: (batch, 2, 1024)           # I/Q signal data
├── Conv1d: 2 → 32 channels       # Robust initial extraction
├── Conv1d: 32 → 64 channels      # Max channel capacity
├── Conv1d: 64 → 64 channels      # Maintain high-level features
├── Conv1d: 64 → 48 channels      # Gradual reduction
├── Conv1d: 48 → 32 channels      # Continue reduction
├── Conv1d: 32 → 16 channels      # Final feature consolidation
└── Classifier: 16*4 → 256 → 128 → 64 → 8 classes
```

**Model Statistics**:
- **Parameters**: ~400K
- **Memory Usage**: ~5.5GB (batch size 256)
- **Target Use**: Higher capacity when 32-channel model plateaus

## Understanding Channel Concepts

### Data Channels vs Model Channels

| Aspect | Data Channels | Model Channels |
|--------|---------------|----------------|
| **Definition** | Input signal components | CNN feature maps |
| **Count** | Always 2 (I/Q) | Variable (16, 32, 64, etc.) |
| **Purpose** | Represent complex signal | Extract different patterns |
| **Physical Meaning** | I/Q components of RF signal | Learned feature detectors |
| **Fixed/Variable** | Fixed by signal format | Designed by architecture |

### Data Channels (Input)
```
I/Q Signal Representation:
- I Channel: Amplitude × cos(phase)
- Q Channel: Amplitude × sin(phase)
- Combined: Represents complex baseband signal
```

### Model Channels (Feature Maps)
```
What Each Model Channel Learns (Examples):
- Channel 1: Low frequency components
- Channel 2: High frequency components  
- Channel 3: Phase transition patterns
- Channel 4: Amplitude modulation patterns
- Channel 5: Symbol timing patterns
- ...
- Channel N: Complex modulation-specific features
```

## Training Configuration

### Hardware Setup
- **GPU**: NVIDIA RTX 5070 Ti (16GB VRAM)
- **Batch Size**: 256
- **Framework**: PyTorch with CUDA

### Data Split
- **Training**: 70% of samples
- **Validation**: 20% of samples  
- **Testing**: 10% of samples

### Current Training Results

#### CNN-32 Channel Model Performance
```
Training Results (100 epochs):
├── Final Training Accuracy: 39.71%
├── Final Validation Accuracy: 41.73%
├── Best Validation Accuracy: 41.73%
└── Training Characteristics:
    ├── Slow initial learning (0-80 epochs: ~35%)
    ├── Late improvement (80-100 epochs: 35% → 42%)
    ├── No severe overfitting (train/val gap: 0-4%)
    └── Stable convergence
```

**Performance Analysis**:
- ❌ **Below Target**: 42% vs expected 70-85%
- ❌ **Slow Learning**: Plateaued for most epochs
- ✅ **No Overfitting**: Good train/val balance
- ✅ **Stable Training**: Consistent convergence

## Technical Challenges Encountered

### 1. Kernel Size Issues
**Problem**: Initial models used `kernel_size=12` with `stride=3`, causing dimension reduction too fast.
```
Error: RuntimeError: Kernel size can't be greater than actual input size
```

**Solution**: Implemented progressive kernel size reduction:
- Early layers: Large kernels (9, 7, 5)
- Later layers: Small kernels (3, 3, 3)

### 2. Performance Issues
**Current Challenge**: 41.73% accuracy is below expected performance for 8-class classification.

**Possible Causes**:
- Model underfitting (insufficient capacity)
- Suboptimal learning rate scheduling
- Lack of data augmentation
- Inadequate feature extraction for I/Q signals

## Model Selection Strategy

### When to Use 32-Channel Model:
- ✅ Baseline experiments
- ✅ Limited computational resources
- ✅ Overfitting concerns
- ✅ Fast prototyping

### When to Use 64-Channel Model:
- ✅ 32-channel model plateaus early
- ✅ Need higher model capacity
- ✅ Sufficient training data available
- ✅ Target higher accuracy

### Performance Comparison Framework:
```
Model Evaluation Metrics:
├── Accuracy (Target: >70% for 8 classes)
├── Training Efficiency (Epochs to convergence)
├── Overfitting Analysis (Train/Val gap)
├── Memory Usage (GPU constraints)
└── Inference Speed (Real-time requirements)
```

## Next Steps & Improvements

### Immediate Priorities:
1. **Data Augmentation Implementation**
   - AWGN noise injection
   - Phase rotation simulation
   - Frequency offset modeling
   - I/Q imbalance simulation

2. **Training Optimization**
   - Learning rate scheduling improvements
   - Better optimizer configuration
   - Focal loss for hard example focus

3. **Architecture Refinements**
   - Multi-scale feature extraction
   - Attention mechanisms
   - Residual connections

Expected Improvements:

## Key Insights Learned

### Channel Architecture Design:
1. **Progressive Channel Reduction**: Start wide, narrow down for classification
2. **Kernel Size Management**: Use large kernels early, small kernels later
3. **Regularization Balance**: More channels = need more regularization

### Signal Processing Considerations:
1. **I/Q Data Handling**: Preserve phase relationships
2. **SNR Robustness**: Model must work across noise conditions
3. **Real-World Deployment**: Hardware imperfections matter

### Training Strategies:
1. **Overfitting vs Underfitting**: Current model shows underfitting
2. **Learning Rate Impact**: Conservative LR may slow convergence
3. **Data Augmentation Necessity**: Critical for RF signal robustness

---

**Status**: Development in progress  
**Current Focus**: Implementing data augmentation for performance improvement  
**Target**: 70-85% validation accuracy for practical deployment
