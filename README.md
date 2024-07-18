# Multilayer-Perceptron

## Overview
This project consists of implementing several scripts for splitting datasets, training, and evaluating a neural network model for classification tasks. We will use those scripts in order to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis in the Wisconsin.

Below is a brief description of each script:

## File Structure
- `mlp.py`: Main file, used to launch either the whole program, or a specific part below using arguments.
- `srcs/`
  - `split.py`: Dataset splitting functions.
  - `train.py`: Training functions.
  - `predict.py`: Model evaluation functions.
  - `histograms.py`: Data visualization.

- **train.py**: Contains functions to run the training process of a neural network model using specified parameters.
- **split.py**: Includes functions to split a dataset into training and testing sets and save them to CSV files.
- **predict.py**: Provides functions to evaluate a trained model on a test dataset and display performance metrics.
- **histograms.py**: Plots the data repartition, using histograms, for each feature. Very useful to understand which features will be the most relevant ones.

## Initialization

1. **Init.sh**: Run this bash script to create a virtual environment and install dependencies located in the `requirements.txt` file.
2. If the `/venv` folder already exists, you will be prompted whether you want to overwrite the existing one or not.

## Usage

1. **Configuring the model**:
   - Using my custom `neural_network_lib`, go to the `/config` folder, you'll find `model_config.yaml`.
   - Create the different layers required for your model, for example: 
   ```yaml
   layers:
   - type: InputLayer
      input_size: 30
   - type: Dense
      input_size: 30
      output_size: 24
      kernel_initializer: xavier
   - type: Sigmoid
   - type: Dense
      input_size: 24
      output_size: 16
      kernel_initializer: xavier
   - type: Sigmoid
   - type: Dense
      input_size: 16
      output_size: 2
      kernel_initializer: xavier
   - type: Softmax
   ```
   This config will be loaded automatically.

2. **Splitting Dataset**:
   - Use `split_dataset(input_csv, test_size, random_state)` in `split.py` to split a dataset and save the train and test sets.
   - You can run for example: 
   `
   python mlp.py split --input_csv data/data.csv --random_state 64
   `

3. **Training the Model**:
   - Run `run_training(epochs, optimizer, learning_rate, early_stopping, patience, batch_size)` in `train.py` to train the model.
   - Ensure the required columns are present in the training and test CSV files.
   - You can run for example: 
   `
   python mlp.py train --learning_rate 0.001 --optimizer sgd --epochs 50 --batch_size 16
   `

4. **Evaluating Model**:
   - Call `evaluate_model(test_csv, activations)` in `predict.py` to evaluate the trained model on a test dataset.
   - You can run for example:
   `
   python mlp.py predict
   `

5. **Run the full pipepline**:
   - You can run for example:  
   `
   python mlp.py full --input_csv data/data.csv --test_csv data/train_test/test.csv --learning_rate 0.001 --optimizer adam --epochs 500 --early_stopping --random_state rand
   `

## Dependencies
- `neural_network_lib`: My custom internal library for neural network models and utilities.
- `colorama`: Library for colored output in the console.
- `pandas` & `numpy`: Libraries for data analysis and manipulation tool.
- `matplotlib`& `seaborn`: Libraries for data visualization.
- And few other libraries...  
   Go to `requirements.txt`in order to see them all.

## Note
Ensure to set up the necessary data files in the `data/` directory before running the scripts.  
Feel free to explore the `neural_network_lib` library, I've documented everything.


## Demo

[![Watch the video](https://img.youtube.com/vi/WTWw2Ai2UWs/maxresdefault.jpg)](https://www.youtube.com/watch?v=WTWw2Ai2UWs)

