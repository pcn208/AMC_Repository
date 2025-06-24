# Research Log

## experiment/data-augmentation

**Date:** 2025-06-24  
**Branch:** experiment/data-augmentation

### Goals
- Integrate data-augmentation into `focusonCNN.ipynb`
- Track I/Q swap, noise, phase-shift, etc.
- Maximize the potential for RTX 5070 Ti vram 16 gb 
- add Augmentation signal before input it to the model 

### Steps Taken
1. Created `augment_iq_signal()` helper in notebook  
2. Configured `params` ranges for AWGN, SNR, phase rotation…  
3. Create 4 model : 
    ### Models in Code
    You have 4 different CNN models in your code:
    - **BaseCNN_NET**  
        Simple CNN with 3 blocks (32→64→128 channels)  
        Created with `create_improved_model()`
    - **CNN32Channels**  
        6-block CNN with max 32 channels  
        Channel progression: 2→16→32→32→24→16→8
    - **CNN64Channels**  
        6-block CNN with max 64 channels  
        Channel progression: 2→32→64→64→48→32→16
    - **ComplexCNN_NET**  
    7-block CNN with residual connections  
    Goes up to 256 channels  
    Created with `create_complex_model`

### Next Actions
- Plot a few augmented examples  
- Tune dropout vs. augmentation mix  
- Branch for CNN-LSTM fusion

