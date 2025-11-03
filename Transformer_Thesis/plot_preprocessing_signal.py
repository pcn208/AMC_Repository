"""
Raw IQ Signal Visualization Script

This script loads and visualizes raw I/Q signal data from the RadioML 2018 dataset.
It displays:
1. Time-domain I and Q channel signals
2. Constellation diagrams (I vs Q)
3. Magnitude and phase representations
4. Signals from different modulation types and SNR levels
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for data visualization"""

    # Paths - Update these to match your system
    FILE_PATH = "C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
    JSON_PATH = 'C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\classes-fixed.json'

    # Output directory
    OUTPUT_DIR = Path("visualization_results")

    # Visualization settings
    NUM_SAMPLES_PER_MOD = 3  # Number of samples to visualize per modulation
    SEQUENCE_LENGTH = 1024

    # Target modulations to visualize
    TARGET_MODULATIONS = [
        'OOK',
        '4ASK',
        '8ASK',
        'BPSK',
        'QPSK',
        '8PSK',
        '16PSK',
        '32PSK',
        '16APSK',
        '32APSK',
        '64APSK',
        '128APSK',
        '16QAM',
        '32QAM',
        '64QAM',
        '128QAM',
        '256QAM',
        'GMSK',
        'OQPSK'
    ]


def load_dataset_info(file_path, json_path):
    """
    Load dataset information and metadata

    Args:
        file_path: Path to HDF5 file
        json_path: Path to JSON class file

    Returns:
        Dictionary containing dataset info
    """
    print(f"📂 Loading dataset from: {file_path}")

    with h5py.File(file_path, 'r') as f:
        # Get dataset dimensions
        X_shape = f['X'].shape
        Y_shape = f['Y'].shape
        Z_shape = f['Z'].shape

        print(f"\nDataset shapes:")
        print(f"  X (IQ data): {X_shape}")
        print(f"  Y (labels):  {Y_shape}")
        print(f"  Z (SNR):     {Z_shape}")

        # Load labels and SNR
        Y_int = np.argmax(f['Y'][:], axis=1)
        Z = f['Z'][:, 0]

    # Load class names
    with open(json_path, 'r') as f:
        modulation_classes = json.load(f)

    Y_strings = np.array([modulation_classes[i] for i in Y_int])

    # Get unique SNR values
    unique_snr = np.unique(Z)

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(Y_strings):,}")
    print(f"  Modulation types: {len(modulation_classes)}")
    print(f"  SNR range: {unique_snr.min():.1f} to {unique_snr.max():.1f} dB")
    print(f"  SNR values: {sorted(unique_snr)}")

    return {
        'file_path': file_path,
        'Y_strings': Y_strings,
        'Z': Z,
        'modulation_classes': modulation_classes,
        'unique_snr': unique_snr
    }


def plot_iq_signal(i_signal, q_signal, title, save_path=None):
    """
    Plot I/Q signal in multiple representations

    Args:
        i_signal: In-phase component (numpy array)
        q_signal: Quadrature component (numpy array)
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Time axis
    time_axis = np.arange(len(i_signal))

    # 1. Time domain - I and Q signals
    ax1 = axes[0, 0]
    ax1.plot(time_axis, i_signal, label='I (In-phase)', alpha=0.7, linewidth=0.8)
    ax1.plot(time_axis, q_signal, label='Q (Quadrature)', alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time Domain - I/Q Signals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Constellation diagram (I vs Q)
    ax2 = axes[0, 1]
    ax2.scatter(i_signal, q_signal, alpha=0.3, s=5)
    ax2.set_xlabel('I (In-phase)')
    ax2.set_ylabel('Q (Quadrature)')
    ax2.set_title('Constellation Diagram')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.axis('equal')

    # 3. Magnitude over time
    ax3 = axes[1, 0]
    magnitude = np.sqrt(i_signal**2 + q_signal**2)
    ax3.plot(time_axis, magnitude, color='purple', linewidth=0.8)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Signal Magnitude')
    ax3.grid(True, alpha=0.3)

    # 4. Phase over time
    ax4 = axes[1, 1]
    phase = np.arctan2(q_signal, i_signal)
    ax4.plot(time_axis, phase, color='orange', linewidth=0.8)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Phase (radians)')
    ax4.set_title('Signal Phase')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")

    plt.close()


def visualize_modulation_samples(file_path, dataset_info, modulation, config, num_samples=3, snr_value=None):
    """
    Visualize samples for a specific modulation type

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        modulation: Modulation type to visualize
        config: Configuration object
        num_samples: Number of samples to visualize
        snr_value: Specific SNR value to filter (optional, uses highest if None)
    """
    Y_strings = dataset_info['Y_strings']
    Z = dataset_info['Z']

    # Filter indices for this modulation
    mod_indices = np.where(Y_strings == modulation)[0]

    if len(mod_indices) == 0:
        print(f"  ⚠️  No samples found for modulation: {modulation}")
        return

    # Filter by SNR if specified, otherwise use highest SNR
    if snr_value is None:
        snr_value = dataset_info['unique_snr'].max()

    snr_mask = np.abs(Z[mod_indices] - snr_value) < 0.1
    filtered_indices = mod_indices[snr_mask]

    if len(filtered_indices) == 0:
        print(f"  ⚠️  No samples found for {modulation} at SNR={snr_value}dB")
        return

    # Select random samples
    num_samples = min(num_samples, len(filtered_indices))
    sample_indices = np.random.choice(filtered_indices, num_samples, replace=False)

    print(f"\n📊 Visualizing {modulation} (SNR={snr_value}dB) - {num_samples} samples")

    # Create output directory for this modulation
    mod_dir = config.OUTPUT_DIR / modulation
    mod_dir.mkdir(parents=True, exist_ok=True)

    # Load and plot each sample
    with h5py.File(file_path, 'r') as f:
        X = f['X']

        for i, idx in enumerate(sample_indices):
            # Load raw IQ data
            x_raw = X[idx]  # Shape: [1024, 2]

            i_signal = x_raw[:, 0]  # In-phase
            q_signal = x_raw[:, 1]  # Quadrature

            # Get actual SNR for this sample
            actual_snr = Z[idx]

            # Create plot
            title = f"{modulation} - Sample {i+1} (SNR={actual_snr:.1f}dB)"
            save_path = mod_dir / f"{modulation}_sample_{i+1}_snr{int(actual_snr)}.png"

            plot_iq_signal(i_signal, q_signal, title, save_path)


def visualize_snr_comparison(file_path, dataset_info, modulation, config, snr_values=None):
    """
    Visualize the same modulation type at different SNR levels

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        modulation: Modulation type to visualize
        config: Configuration object
        snr_values: List of SNR values to compare (optional)
    """
    if snr_values is None:
        # Select low, medium, and high SNR
        unique_snr = dataset_info['unique_snr']
        snr_values = [unique_snr.min(), unique_snr[len(unique_snr)//2], unique_snr.max()]

    Y_strings = dataset_info['Y_strings']
    Z = dataset_info['Z']

    print(f"\n📊 SNR Comparison for {modulation}")

    # Create comparison directory
    comp_dir = config.OUTPUT_DIR / "snr_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, 'r') as f:
        X = f['X']

        fig, axes = plt.subplots(len(snr_values), 3, figsize=(16, 4*len(snr_values)))
        if len(snr_values) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{modulation} - SNR Comparison', fontsize=16, fontweight='bold')

        for row, snr in enumerate(snr_values):
            # Find sample at this SNR
            mod_indices = np.where(Y_strings == modulation)[0]
            snr_mask = np.abs(Z[mod_indices] - snr) < 0.1
            filtered_indices = mod_indices[snr_mask]

            if len(filtered_indices) == 0:
                print(f"  ⚠️  No samples found at SNR={snr}dB")
                continue

            # Get one sample
            idx = filtered_indices[0]
            x_raw = X[idx]

            i_signal = x_raw[:, 0]
            q_signal = x_raw[:, 1]
            time_axis = np.arange(len(i_signal))

            # Plot I/Q time series
            axes[row, 0].plot(time_axis, i_signal, label='I', alpha=0.7, linewidth=0.8)
            axes[row, 0].plot(time_axis, q_signal, label='Q', alpha=0.7, linewidth=0.8)
            axes[row, 0].set_ylabel(f'SNR={snr:.1f}dB', fontsize=12, fontweight='bold')
            axes[row, 0].legend()
            axes[row, 0].grid(True, alpha=0.3)
            if row == 0:
                axes[row, 0].set_title('Time Domain')
            if row == len(snr_values) - 1:
                axes[row, 0].set_xlabel('Sample Index')

            # Plot constellation
            axes[row, 1].scatter(i_signal, q_signal, alpha=0.3, s=5)
            axes[row, 1].axhline(y=0, color='k', linewidth=0.5)
            axes[row, 1].axvline(x=0, color='k', linewidth=0.5)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].axis('equal')
            if row == 0:
                axes[row, 1].set_title('Constellation Diagram')
            if row == len(snr_values) - 1:
                axes[row, 1].set_xlabel('I (In-phase)')
            axes[row, 1].set_ylabel('Q (Quadrature)')

            # Plot magnitude
            magnitude = np.sqrt(i_signal**2 + q_signal**2)
            axes[row, 2].plot(time_axis, magnitude, color='purple', linewidth=0.8)
            axes[row, 2].grid(True, alpha=0.3)
            if row == 0:
                axes[row, 2].set_title('Magnitude')
            if row == len(snr_values) - 1:
                axes[row, 2].set_xlabel('Sample Index')
            axes[row, 2].set_ylabel('Magnitude')

        plt.tight_layout()
        save_path = comp_dir / f"{modulation}_snr_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
        plt.close()


def create_overview_plot(file_path, dataset_info, config):
    """
    Create an overview plot showing multiple modulation types

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        config: Configuration object
    """
    print(f"\n📊 Creating overview plot")

    # Select a subset of modulations for overview
    overview_mods = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'GMSK']
    available_mods = [m for m in overview_mods if m in config.TARGET_MODULATIONS]

    if len(available_mods) == 0:
        print("  ⚠️  No modulations available for overview")
        return

    Y_strings = dataset_info['Y_strings']
    Z = dataset_info['Z']

    # Use highest SNR for clearest signals
    snr_value = dataset_info['unique_snr'].max()

    with h5py.File(file_path, 'r') as f:
        X = f['X']

        fig, axes = plt.subplots(len(available_mods), 2, figsize=(14, 3*len(available_mods)))
        if len(available_mods) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Modulation Overview (SNR={snr_value:.1f}dB)', fontsize=16, fontweight='bold')

        for row, mod in enumerate(available_mods):
            # Find sample
            mod_indices = np.where(Y_strings == mod)[0]
            snr_mask = np.abs(Z[mod_indices] - snr_value) < 0.1
            filtered_indices = mod_indices[snr_mask]

            if len(filtered_indices) == 0:
                continue

            idx = filtered_indices[0]
            x_raw = X[idx]

            i_signal = x_raw[:, 0]
            q_signal = x_raw[:, 1]
            time_axis = np.arange(len(i_signal))

            # Time domain
            axes[row, 0].plot(time_axis, i_signal, label='I', alpha=0.7, linewidth=0.8)
            axes[row, 0].plot(time_axis, q_signal, label='Q', alpha=0.7, linewidth=0.8)
            axes[row, 0].set_ylabel(mod, fontsize=12, fontweight='bold')
            axes[row, 0].legend()
            axes[row, 0].grid(True, alpha=0.3)
            if row == 0:
                axes[row, 0].set_title('Time Domain')
            if row == len(available_mods) - 1:
                axes[row, 0].set_xlabel('Sample Index')

            # Constellation
            axes[row, 1].scatter(i_signal, q_signal, alpha=0.3, s=5)
            axes[row, 1].axhline(y=0, color='k', linewidth=0.5)
            axes[row, 1].axvline(x=0, color='k', linewidth=0.5)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].axis('equal')
            if row == 0:
                axes[row, 1].set_title('Constellation Diagram')
            if row == len(available_mods) - 1:
                axes[row, 1].set_xlabel('I (In-phase)')
            axes[row, 1].set_ylabel('Q (Quadrature)')

        plt.tight_layout()
        save_path = config.OUTPUT_DIR / "modulation_overview.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
        plt.close()


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='Visualize Raw IQ Signals')
    parser.add_argument('--file_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--json_path', type=str, help='Path to JSON file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--modulations', type=str, nargs='+', help='Specific modulations to visualize')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples per modulation')
    parser.add_argument('--create_overview', action='store_true', help='Create overview plot')
    parser.add_argument('--snr_comparison', action='store_true', help='Create SNR comparison plots')

    args = parser.parse_args()

    # Update config from args
    config = Config()
    if args.file_path:
        config.FILE_PATH = args.file_path
    if args.json_path:
        config.JSON_PATH = args.json_path
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    if args.num_samples:
        config.NUM_SAMPLES_PER_MOD = args.num_samples

    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("RAW IQ SIGNAL VISUALIZATION")
    print("="*70)

    # Load dataset info
    dataset_info = load_dataset_info(config.FILE_PATH, config.JSON_PATH)

    # Determine which modulations to visualize
    if args.modulations:
        modulations_to_viz = args.modulations
    else:
        modulations_to_viz = config.TARGET_MODULATIONS[:5]  # First 5 by default

    print(f"\nModulations to visualize: {modulations_to_viz}")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create overview plot
    if args.create_overview or not args.modulations:
        create_overview_plot(config.FILE_PATH, dataset_info, config)

    # Visualize each modulation
    for modulation in modulations_to_viz:
        if modulation in config.TARGET_MODULATIONS:
            visualize_modulation_samples(
                config.FILE_PATH,
                dataset_info,
                modulation,
                config,
                num_samples=config.NUM_SAMPLES_PER_MOD
            )

            # Create SNR comparison if requested
            if args.snr_comparison:
                visualize_snr_comparison(
                    config.FILE_PATH,
                    dataset_info,
                    modulation,
                    config
                )
        else:
            print(f"  ⚠️  Skipping unknown modulation: {modulation}")

    print("\n" + "="*70)
    print(f"✅ Visualization complete! Results saved to: {config.OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
