# Compound Eye Classification

This project focuses on the classification of compound eye images using machine learning techniques.

## Environment
* Python 3.7.13
* Pytorch 1.12.1
These are the recommendation package version to reproduce the experiment. The program can still run with different package versions, but results may not be the same.

## Usage
To train the model, run:
```bash
python main.py --data_path /path/to/dataset
```
Replace `/path/to/dataset` with the actual location of the dataset folder. For example, if the dataset folder and `main.py` are in the same directory, use
```bash
python main.py --data_path ./
```

## Results
The results of the model training and evaluation will be saved in the `results/` directory. This includes accuracy, loss plots, and confusion matrices.
