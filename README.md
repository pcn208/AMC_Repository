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

### New Log 

**Date:** 2025-06-30 

### Improvement Model : 
/home_notebook/CNN_IQ.py 

**Structure:** 
| Component            | Type / Layer                                   | Output Shape               | Description                                                      |
|----------------------|-------------------------------------------------|----------------------------|------------------------------------------------------------------|
| **CNNIQBranch**      |                                                 |                            | *Processes one channel (I or Q) through two conv blocks + GAP*   |
| `conv1`              | Conv2d(1 → 64, 3×3, pad=1)                      | (batch, 64, H, W)          | 1st convolution                                                 |
| `bn1`                | BatchNorm2d(64)                                 | (batch, 64, H, W)          | Normalizes after `conv1`                                        |
| `conv1_2`            | Conv2d(64 → 64, 3×3, pad=1)                     | (batch, 64, H, W)          | 2nd convolution in block 1                                      |
| `bn1_2`              | BatchNorm2d(64)                                 | (batch, 64, H, W)          | Normalizes after `conv1_2`                                      |
| `pool` + `dropout`   | MaxPool2d(3×3,stride=2,pad=1) + Dropout2d        | (batch, 64, ⌈H/2⌉, ⌈W/2⌉)   | Downsamples + regularizes                                       |
| `conv2`              | Conv2d(64 → 128, 3×3, pad=1)                    | (batch, 128, ⌈H/2⌉, ⌈W/2⌉) | 1st convolution in block 2                                       |
| `bn2`                | BatchNorm2d(128)                                | (batch, 128, ⌈H/2⌉, ⌈W/2⌉) | Normalizes after `conv2`                                        |
| `conv2_2`            | Conv2d(128 → 128, 3×3, pad=1)                   | (batch, 128, ⌈H/2⌉, ⌈W/2⌉) | 2nd convolution in block 2                                      |
| `bn2_2`              | BatchNorm2d(128)                                | (batch, 128, ⌈H/2⌉, ⌈W/2⌉) | Normalizes after `conv2_2`                                      |
| `pool` + `dropout`   | MaxPool2d + Dropout2d                           | (batch, 128, ⌈H/4⌉, ⌈W/4⌉) | Further downsampling + dropout                                  |
| `global_avg_pool`    | AdaptiveAvgPool2d(1×1)                          | (batch, 128, 1, 1)         | Collapses spatial dims to 1×1                                   |
| `flatten`            | —                                               | (batch, 128)               | Flattens for fully-connected layers                             |
| **CNNIQModel**       |                                                 |                            | *Fuses I/Q branches and classifies*                             |
| `i_branch`, `q_branch` | CNNIQBranch(dropout×0.75)                    | (batch, 128)               | Separate feature extractors for I and Q                         |
| `combined_features`  | Element-wise add                                 | (batch, 128)               | Simple addition fusion of I & Q features                        |
| **Classifier**       |                                                 |                            | *Three dense layers + output*                                   |
| `Linear(128→256)`    | Linear + BatchNorm1d + LeakyReLU + Dropout       | (batch, 256)               | Expands feature dim                                            |
| `Linear(256→128)`    | Linear + BatchNorm1d + LeakyReLU + Dropout       | (batch, 128)               | Reduces dim with normalization                                  |
| `Linear(128→64)`     | Linear + LeakyReLU + Dropout                     | (batch, 64)                | Further reduction                                              |
| `Linear(64→num_classes)` | Linear                                     | (batch, num_classes)       | Final logits for classification                                 |

### TARGET MODULATION : 
    ['OOK','4ASK','8ASK','BPSK', 'QPSK', '8PSK', '16QAM','64QAM']
    Total dataset size: 49504
    Train dataset size: 49504 (80%)
    Validation dataset size: 14144 (20%)

### CONFUSION MATRIX 
    ![Confusion matrix plot showing classification results for eight modulation types: OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16QAM, and 64QAM. The matrix displays true labels on the vertical axis and predicted labels on the horizontal axis, with cell values indicating the number of samples for each true-predicted pair. Most values are concentrated along the diagonal, indicating correct predictions, with some off-diagonal values showing misclassifications. The overall accuracy is 66.38 percent. A blue color gradient bar on the right represents the count scale, with darker shades indicating higher values. The environment is a scientific or research context, focusing on evaluating model performance. The following text appears at the top: Confusion Matrix I signal and Q signal process separately Best Epoch Overall Accuracy 66.38 percent. The axes are labeled True Label and Predicted Label.](home_notebook/best_confusion_matrix.png)
### Heatmap Overall Accuracy Each Modulation with each SNR level 
    ![Heatmap plot showing classification accuracy for each target modulation type across different SNR levels. The x-axis lists SNR values, the y-axis lists modulation types: OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16QAM, and 64QAM. Each cell displays the accuracy for a specific modulation and SNR, with a color gradient indicating accuracy levels—darker shades represent higher accuracy. The environment is a research setting, visualizing model performance across conditions.](home_notebook/modulation_snr_accuracy_heatm.png)
