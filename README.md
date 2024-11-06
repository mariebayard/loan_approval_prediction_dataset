# Loan Approval Prediction Application

This project is part of a supervised machine learning course. Using a Kaggle dataset, it aims to predict whether a loan will be approved based on the applicant’s financial and personal information. The application allows users to input values and generate predictions using various machine learning models.

## Table of Contents
- [Overview](#overview)
- [Data Processing](#data-processing)
- [Visualization](#visualization)
- [Statistical Analysis](#statistical-analysis)
- [Modeling](#modeling)
- [Flask Application](#flask-application)
- [Installation](#installation)
- [Results and Conclusions](#results-and-conclusions)

## Overview

The dataset includes the following columns:

- **Variables**: `loan_id`, `no_of_dependants`, `education`, `self_employed`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_asset_value`, `bank_asset_value`, `loan_status`.
- **Data Types**: `education`, `self_employed`, and `loan_status` are categorical, while all others are numerical. No missing values were found in the dataset.

## Data Processing

Several data-cleaning steps were performed to prepare the dataset for modeling:

- Removed the `loan_id` variable
- Encoded categorical variables (`education`, `self_employed`, `loan_status`) as binary values
- Checked for duplicates; none were found

## Visualization

Exploratory data analysis was conducted using the following visualizations:

- **Correlation Heatmap**
- **Distribution Plot** for income and loan amount
- **Boxplots** of `loan_amount`, `income_annum`, `cibil_score` by `loan_status`
- **Jointplot** between `cibil_score` and `loan_status`
- **FacetGrid** to view `cibil_score` distribution by `loan_status`

## Statistical Analysis

To validate relationships in the data, the following statistical tests were applied:

- **Mann-Whitney U** and **Shapiro-Wilk** tests for `loan_status` against `loan_amount`, `income_annum`, and `cibil_score`
- **Kruskal-Wallis** and **Shapiro-Wilk** tests for `no_of_dependants` and `loan_amount`, as well as `loan_term` and `loan_status`

## Modeling

The following models were tested and evaluated:

- **Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, and ANN
- **Optimization**: Hyperparameters were optimized using GridSearch
- **Feature Engineering**: Scaling, balancing, and new features were tested in the ANN models to improve performance
- **Best Models**: Random Forest and XGBoost achieved 98% accuracy

## Flask Application

A Flask application was built to allow interactive access to the trained models. Key features include:

- **Predictions** based on user inputs
- **Warnings** if input values fall outside of the training data ranges
- **User-friendly design** with CSS and HTML templates

## Installation

1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. Run the application locally using python app.py.

## Results and Conclusions

The best-performing models were Random Forest and XGBoost, each achieving 98% accuracy. Both models identified cibil_score as the most significant factor for loan approval. Decision Tree and ANN also performed well, with approximately 97% accuracy. Feature engineering and hyperparameter tuning further enhanced the models’ performance.
