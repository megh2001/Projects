# ResNet Model for CIFAR-10 Classification

Kaggle link: https://www.kaggle.com/competitions/deep-learning-spring-2025-project-1/leaderboard?
Team name: Pomodoro, placed 5th

This repository contains our implementation for the Deep Learning Spring 2025 Project 1, where we develop a modified ResNet architecture to achieve high accuracy on the CIFAR-10 dataset while staying under 5 million parameters.


## Project Overview

We've implemented a Wide ResNet variant that balances model size and performance on the CIFAR-10 image classification task. Our model:
- Uses a parameter-efficient ResNet design with 3.38 million parameters
- Implements residual blocks with batch normalization
- Incorporates modern training techniques including mixup and label smoothing

## Repository Structure

- 'Google_Collab_notebook.ipynb' : Contains all the code related to the project. You can either run this Google Colab notebook or execute the following scripts in order: train.py, generate_csv.py, and run_validation.py.
- `train.py`: Script to train the model from scratch
- `generate_csv.py`: Script to generate predictions on test data and generate csv file 
-  'run_validation.py' : Script to generate accuracy,confusion matrix and other analysis validation dataset.
- `best_model.pth`: Our best performing model weights
- `requirements.txt`: Required dependencies
- `README.md`: Project documentation
- 'Results' : Contains images related to results

## Model Architecture

Our model is a WideResNet with:
- 3 stages of residual blocks with [3, 3, 3] layers per stage
- Width multiplier of 1.0 to control model size
- Dropout rate of 0.2 for regularization
- Total parameters: ~4.38 million

## Training Methodology

We used the following training strategy:
- SGD optimizer with momentum 0.9
- Cosine annealing learning rate schedule
- Initial learning rate of 0.1
- Weight decay of 5e-4
- 1000 epochs of training
- Mixup data augmentation with alpha=0.2
- Label smoothing with value 0.1

## Data Augmentation

For training data, we applied multiple augmentation techniques:
- Random cropping with padding
- Random horizontal flips
- Random rotations
- Color jittering
- Random erasing

## Results

Our model achieves:
- Validation accuracy: 97.09%
- Test accuracy: 88.913%

## How to Run

1. Upload the data(deep-learning-spring-2025-project-1) to your working directory. 
2. Either run google collab notebook (which has all the necessary code and is recommeded) or run below codes

```bash
pip install -r requirements.txt
python train.py
python generate_csv.py
pyhton run_validation.py
```


## Citation
Our implementation was inspired by:
- [ResNet paper](https://arxiv.org/abs/1512.03385)
- [Wide ResNet paper](https://arxiv.org/abs/1605.07146)
