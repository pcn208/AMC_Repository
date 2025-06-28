# RadioML18 CNN-IQ Training Results Log

## 📊 Current Best Result: **58.65% Validation Accuracy**

**Date:** December 2024  
**Early Stopping:** Epoch 56  
**Dataset:** RadioML18 (GOLD_XYZ_OSC.0001_1024.hdf5)

---

## 🎯 **Target Modulations (7 classes)**
```python
TARGET_MODULATIONS = ['OOK','4ASK', 'BPSK', 'QPSK', '8PSK','32PSK', '16QAM']
```

## ⚙️ **Best Working Configuration**

### Model Architecture
- **Model:** CNNIQModel (dual-branch CNN for I/Q processing)
- **Dropout:** 0.3
- **Fusion:** Addition-based I/Q feature combination
- **Classes:** 7

### Training Parameters
```python
BATCH_SIZE = 128
NUM_EPOCHS = 300
patience = 25
INPUT_CHANNELS = 2
SEQUENCE_LENGTH = 1024
```

### Optimizer & Scheduler
```python
optimizer = optim.AdamW(
    model_CNNIQ.parameters(), 
    lr=0.0015,
    weight_decay=0.005,
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[50, 100, 150], 
    gamma=0.5
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Signal Preprocessing
**Key breakthrough:** Amplitude/Phase representation instead of raw I/Q
```python
# Normalize signals first
i_norm = i_signal / (torch.sqrt(i_signal**2 + q_signal**2 + 1e-8))
q_norm = q_signal / (torch.sqrt(i_signal**2 + q_signal**2 + 1e-8))

# Create amplitude and phase
amplitude = torch.sqrt(i_signal**2 + q_signal**2)
phase = torch.atan2(q_signal, i_signal)

# Reshape to 32x32 for CNN
i_2d = amplitude.view(1, 32, 32)
q_2d = phase.view(1, 32, 32)
```

---

## 📈 **Accuracy Progression**

| Attempt | Accuracy | Key Changes | Notes |
|---------|----------|-------------|-------|
| 1 | 35-43% | Raw I/Q, basic reshape | Baseline |
| 2 | 34% | Added aggressive augmentation | **Failed** - augmentation too strong |
| 3 | 43% | Removed augmentation, tuned hyperparameters | Back on track |
| 4 | 60.78% | **Amplitude/phase preprocessing** | **Major breakthrough** |
| 5 | 54.03% | Increased LR, reduced dropout | **Failed** - unstable training |
| 6 | **58.65%** | Reverted config, extended training | Current best |

---

## 🎯 **Per-Modulation Performance Analysis**

### ✅ **Strong Performers (90-100% at high SNR)**
- **OOK:** Excellent performance across all SNRs
- **4ASK:** Consistent high performance  
- **BPSK:** Strong, reaches 100% at high SNR
- **QPSK:** Good performance, 87.5-100% at mid-high SNR
- **16QAM:** Outstanding, 100% across most conditions
- **32PSK:** Surprisingly strong for complex modulation

### ❌ **Main Bottleneck**
- **8PSK:** Maximum 62.5% accuracy, lots of confusion
  - **Root cause:** Likely phase discrimination issues in preprocessing
  - **Impact:** Primary blocker preventing 70%+ accuracy

---

## 🔧 **Hardware Setup**
- **GPU:** RTX 3050 (4GB VRAM)
- **RAM:** 16GB
- **Batch Size:** Limited by GPU memory

---

## 💡 **Key Learnings**

### What Worked ✅
1. **Amplitude/phase preprocessing** - Single biggest improvement (+17% accuracy)
2. **No data augmentation** - RF signals too sensitive to modification
3. **Label smoothing (0.1)** - Helped with overconfident predictions
4. **AdamW optimizer** - Better than standard Adam
5. **Conservative learning rates** - 0.0015 worked better than higher rates

### What Failed ❌
1. **Data augmentation** - Noise, scaling, rotation all hurt performance
2. **Arbitrary 32×32 reshape** - Destroyed signal structure
3. **High learning rates** - Caused training instability
4. **Very low dropout** - Led to overfitting

### Critical Insights 🧠
- **Signal preprocessing is everything** - Model architecture is less important than input representation
- **8PSK discrimination** is the main challenge - needs specialized preprocessing
- **SNR gradient** follows expected RF patterns (low→high accuracy with increasing SNR)

---

## 🚀 **Next Steps**

### Priority 1: Fix 8PSK Performance
- [ ] Try different phase normalization strategies
- [ ] Experiment with constellation-aware preprocessing
- [ ] Test raw I/Q with better temporal preservation

### Priority 2: Training Optimization
- [ ] Reduce batch size to 64 for better gradients
- [ ] Try cosine annealing schedule
- [ ] Experiment with different dropout patterns

### Priority 3: Architecture Exploration
- [ ] Test attention-based I/Q fusion
- [ ] Try 1D CNN approach preserving temporal structure
- [ ] Experiment with deeper networks

---

## 📊 **Target: 80% Accuracy**
**Current:** 58.65%  
**Gap:** 21.35%  
**Main blocker:** 8PSK classification  
**Strategy:** Focus on signal preprocessing improvements

---

## 🔗 **Files**
- Model: `CNN_IQ.py`
- Training: `CNN_two_branch.ipynb` 
- Dataset: RadioML18 (GOLD_XYZ_OSC.0001_1024.hdf5)
- Best weights: `best_cnn_model_IQ.pth`
