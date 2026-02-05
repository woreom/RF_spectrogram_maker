# RF Spectrogram Maker

This repository contains tools for processing Radio Frequency (RF) data files (specifically `.dat` files containing complex IQ samples) to generate spectrogram datasets. It is designed to facilitate the creation of datasets for machine learning applications, such as drone detection or RF signal classification, by converting raw signal data into visual spectrogram representations and adding synthetic noise for data augmentation.

## Features

- **Raw Data Processing**: Efficiently loads and processes `.dat` files (complex64 format).
- **Spectrogram Generation**: Converts time-domain RF signals into frequency-domain spectrograms using STFT (Short-Time Fourier Transform).
- **Data Augmentation**: Supports adding Additive White Gaussian Noise (AWGN) to signals to create datasets with varying Signal-to-Noise Ratios (SNR).
- **Dataset Organization**: Automatically structures output datasets into folders based on SNR levels (e.g., `original`, `0db`, `10db`).
- **Metadata Management**: Generates JSON metadata for each processed sample.
- **Visualizations**: Saves spectrogram images for visual inspection.

## Project Structure

- `main.py`: The entry point of the application. configured to iterate over source directories, apply SNR variations, and call the dataset generation logic.
- `dataset.py`: Handles the orchestration of dataset creation, including file traversing, metadata logging (`.json`), and delegating signal processing tasks.
- `processing.py`: Core signal processing library. Contains functions for:
    - Loading `.dat` files.
    - Computing STFT and spectrograms.
    - Plotting signals.
    - Adding AWGN noise.
- `constants.py`: Central configuration file for global constants such as sampling rate, FFT size, and center frequency.
- `requirements.txt`: List of Python dependencies.

## Installation

1.  Clone the repository.
2.  Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Configuration

You can customize the signal processing parameters in `constants.py`:

```python
FFT_SIZE = 1024          # Size of the FFT window
SAMPLING_RATE = 20000000 # 20 MHz
CENTER_FREQ = 2.447e9    # 2.447 GHz
NUM_FFT_SPEC = 1500      # Number of FFT specifications (usage depends on logic)
RECORDING_TIME = 15      # Expected recording time in seconds
```

## Usage

To generate a dataset, run `main.py`.

**Note:** You will likely need to adjust the `base_path` and `dataset_path` variables in `main.py` to point to your specific data locations.

```python
# Example configuration in main.py
if __name__ == "__main__":
    snr_range = [] # Add specific user defined SNRs if needed, e.g. range(-20, 21, 5)
    base_folder = '2025-12-12'
    
    # Update these paths to match your local directory structure
    dataset_path = f'../../dataset/SiteSurveyRecord/{base_folder}/'
    base_path = f'../../dataset/SiteSurveyRecord/{base_folder}/'

    for folder_name in os.listdir(base_path):
        main(snr_range,
            dataset_path=os.path.join(dataset_path, folder_name),
            base_path=os.path.join(base_path, folder_name), 
            add_original=True)
```

Run the script:

```bash
python main.py
```

### Main Parameters

The `main` function accepts several parameters to control validation and output:

- `snr_range`: A list or range of integer dB values. For each value, a copy of the dataset with added noise will be created.
- `dataset_path`: Directory where the processed dataset will be saved.
- `base_path`: Directory containing the source `.dat` files.
- `patterns`: Glob patterns to match source files (default: `["**/*.dat"]`).
- `add_original`: Boolean flag to determine if the unmodified (original) data should also be saved.

## Output Structure

The script generates a directory structure similar to this:

```
dataset_output_directory/
├── original/
│   ├── sample1_spectrogram.png
│   ├── sample1_meta.json
│   └── ...
├── 0db/
│   ├── sample1_spectrogram.png
│   ├── sample1_meta.json  # Metadata noting the SNR level
│   └── ...
└── 10db/
    └── ...
```
