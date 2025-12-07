# GT-Access
This repository contains the data and code for the paper: Learning User–Resource Interactions for Dynamic Access Control based on Graph–Transformer Fusion, accepted by the 6th International Conference on Social Computing (ICSC 2025).

## Repository Structure
```text
.
├── data/                         # Preprocessed dataset files
├── logs/                         # Training/evaluation logs and outputs
├── DrawUserHistoryLengthHistgram.py
├── GT-Access.py                  # Main entry: training & evaluation (Tables 1–3)
├── models.py                     # Model definitions used in the paper
├── prepare_dataset.py            # Data preparation pipeline for later experiments
├── requirements.txt              # Environment
└── utility.py                    # Utility/helper functions
```
## Quick Start (Reproduce Results)
1) Generate Fig. 2 (User History Length Distribution)
   ```text
   python DrawUserHistoryLengthHistgram.py
   ```
   This will generate Fig. 2: distribution of user access history lengths in the Amazon dataset (saved to the default output path defined in the script).
2) Prepare Dataset
   ```text
   python prepare_dataset.py
   ```
   This will prepare intermediate data files for later experiments (output written under data/ or the configured directory in the script).
3) Run GT-Access to Obtain Tables 1–3
   ```text
   python GT-Access.py
   ```
   This will train/evaluate GT-Access and output the results corresponding to Tables 1, 2, and 3.
   

