# Research Log

## experiment/data-augmentation

**Date:** 2025-06-24  
**Branch:** experiment/data-augmentation

### Goals
- Integrate data-augmentation into `focusonCNN.ipynb`
- Track I/Q swap, noise, phase-shift, etc.

### Steps Taken
1. Created `augment_iq_signal()` helper in notebook  
2. Configured `params` ranges for AWGN, SNR, phase rotation…  
3. …

### Preliminary Observations
- Baseline (no aug): 41.7% acc  
- With I/Q swap only: XX.X% acc  
- …

### Next Actions
- Plot a few augmented examples  
- Tune dropout vs. augmentation mix  
- Branch for CNN-LSTM fusion

