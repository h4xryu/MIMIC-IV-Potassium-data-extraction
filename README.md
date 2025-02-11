# MIMIC-IV ECG and Potassium Analysis

This project analyzes the relationship between ECG waveforms and serum potassium levels using the MIMIC-IV and MIMIC-IV-ECG databases. It matches ECG recordings with nearby potassium measurements and extracts relevant features for analysis.

## Prerequisites

- Python 3.8+
- Access to MIMIC-IV database
- Access to MIMIC-IV-ECG database

## Required Packages

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- wfdb
- tqdm
- jupyter

## Project Structure

```
mimic_analysis/
├── notebooks/
│   └── MIMIC-IV ECG and Potassium Analysis.ipynb
├── data/
│   ├── raw/           # Place MIMIC-IV data here
│   └── processed/     # Processed data will be saved here
├── requirements.txt
└── README.md
```

## Setup

1. Get access to MIMIC-IV and MIMIC-IV-ECG databases from PhysioNet
2. Download the required datasets:
   - MIMIC-IV core module
   - MIMIC-IV hosp module
   - MIMIC-IV-ECG database
3. Update the paths in the notebook to match your local setup:
   ```python
   mimic_path = "path/to/mimiciv/1.0"
   ecg_path = "path/to/mimic-iv-ecg/1.0"
   ```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "MIMIC-IV ECG and Potassium Analysis.ipynb"
```

2. Follow the notebook sections:
   - Data loading and preprocessing
   - ECG and potassium measurement matching
   - Analysis and visualization
   - Results saving

## Features

- Automatic matching of ECG recordings with potassium measurements
- Time-based matching within configurable thresholds
- Potassium level classification (hypokalemia, normal, hyperkalemia)
- Hospital mortality tracking
- Multi-lead ECG waveform processing
- Comprehensive data quality checks

## Data Processing

The system performs the following steps:
1. Loads clinical data from MIMIC-IV CSV files
2. Processes potassium measurements and adds labels
3. Matches ECG recordings with potassium measurements based on time
4. Extracts and processes 12-lead ECG waveforms
5. Saves matched data for further analysis

## Output Format

The matched data is saved in a numpy array with the following structure for each record:
```python
{
    'subject_id': int,
    'record_path': str,
    'Potassium (serum)': float,
    'k_level_label': str,
    'k_level_numeric': int,
    'mort_hosp': int,
    'time_diff_hours': float,
    'signals': numpy.ndarray,
    'fs': float,
    'leads': list,
    'lab_time': datetime,
    'ecg_time': datetime
}
```

## Notes

- Ensure you have sufficient disk space for the MIMIC-IV datasets
- The matching process may take significant time depending on the dataset size
- Consider the time threshold parameter based on your research requirements

## License

This project is intended for research purposes only. Use of MIMIC-IV data requires appropriate credentialing through PhysioNet.
