# Fraud-Detection-in-Minority-Class-of-Credit-Card-Transactions
Build a classification model to predict whether a transaction is fraudulent or not.
This repository contains the code and documentation for a Credit Card Fraud Detection project. The primary goal is to build a robust model that can identify fraudulent transactions from a highly imbalanced dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection](#model-selection)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud detection is a critical issue in the financial industry. This project aims to identify fraudulent transactions from a dataset with a significant class imbalance, where fraudulent transactions represent only 0.17% of the data.

## Dataset

The dataset used for this project is the *Credit Card Fraud Detection* dataset, which contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with 99.83% of the transactions being legitimate and only 0.17% fraudulent.

- *Number of transactions:* 284,807
- *Number of fraudulent transactions:* 492
- *Features:* 30 (including Time, Amount, and 28 anonymized features)

## Project Workflow

1. *Exploratory Data Analysis (EDA):* Understand the distribution of the data, check for missing values, and analyze the imbalance in the dataset.
2. *Data Preprocessing:* Handle missing values, scale features, and split the dataset into training and testing sets.
3. *Feature Engineering:* Create new features, if necessary, and select the most relevant ones.
4. *Modeling:* 
    - *Baseline Models:* Train baseline models to establish a reference point.
    - *Sampling Techniques:* Implement oversampling and undersampling techniques during cross-validation to handle the class imbalance.
    - *Model Evaluation:* Evaluate models using metrics like F1-score, precision, recall, and confusion matrix.
5. *Results:* Present the best-performing model and analyze its strengths and weaknesses.
6. *Future Work:* Identify potential areas for improvement and further experimentation.

## Installation

To run the project locally, clone the repository and install the required dependencies:

bash
git clone https://github.com/Sharjeel862/Fraud-Detection-in-Minority-Class-of-Credit-Card-Transactions/
cd credit-card-fraud-detection
pip install -r requirements.txt


## Usage

To train and evaluate the models, run the following command:

bash
python main.py


This will execute the entire workflow, from data preprocessing to model evaluation.

## Modeling

### Feature Engineering

- *Scaling:* Features are scaled using StandardScaler.
- *Dimensionality Reduction:* Consider using PCA if necessary to reduce the number of features.
- *Feature Selection:* Use techniques like recursive feature elimination (RFE) or feature importance from models.

### Model Selection

We experiment with several models, including:

- Logistic Regression
- Decision Trees
- Gradient Boosting

### Model Evaluation

Given the imbalanced nature of the dataset, we focus on the following metrics:

- *F1-score*
- *Precision*
- *Recall*
- *Confusion Matrix*

StratifiedKFold is used for cross-validation to ensure that each fold has a representative ratio of classes.

## Results

The best model achieved an F1-score of 0.84, with precision and recall of 0.74 and 0.97, respectively. Detailed results can be found in the results directory.

## Future Work

- *Model Improvements:* Explore more advanced techniques such as Hyperparameter Tuning and Cross Validation or IsolationForest Method.
- *Data Augmentation:* Experiment with synthetic data generation to balance the dataset further.
- *Real-Time Detection:* Implement the model in a real-time setting for continuous fraud detection.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
