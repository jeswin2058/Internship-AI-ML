#Loan Approval Prediction using Machine Learning

This project focuses on predicting loan approval status using machine learning models trained on applicant demographic, financial, and credit-related data.
The goal is to assist financial institutions in accurate and faster decision-making by estimating whether a loan application is likely to be approved.

##Project Overview

This project implements a complete end-to-end machine learning workflow:

Import and inspect dataset

Exploratory Data Analysis (EDA)

Data preprocessing & handling missing values

Label encoding of categorical features

Data visualization

Model training using multiple ML algorithms

Model evaluation on unseen test data

This work was completed as part of an internship project under Enterprise Building Training Solutions (EBTS).

##Organization Overview

Enterprise Building Training Solutions (EBTS) is a nationwide training and project-execution organization specializing in:

Data Science

Machine Learning

Software Engineering

Their programs emphasize hands-on, mentor-guided learning, with a strong focus on documentation, reproducibility, and practical implementation.

This internship was carried out under the mentorship of Mr. Arunjit Chowdhury (CEO, EBTS).

##Problem Statement

Build a predictive system that classifies whether a loan application will be:

Approved (Y)

Not Approved (N)

using historical loan application data containing financial and demographic attributes.

##Tech Stack & Tools
###Hardware

Intel Core i7-12650H @ 2.30 GHz

16 GB RAM

GPU: 6 GB

###Software & Libraries

Python 3.8+

pandas, NumPy – Data processing

matplotlib, seaborn – Visualization

scikit-learn – Machine Learning

Jupyter Notebook

##Project Workflow
1. Import Libraries & Load Dataset

Dataset loaded using pandas

Schema validation & datatype checks

Initial exploratory analysis

2. Data Preprocessing

Dropped irrelevant columns (e.g., Loan_ID)

Filled missing values using column means

Ensured dataset consistency

3. Data Encoding

Applied Label Encoding to convert categorical columns to numeric

Ensured final dataset contains only numeric features

4. Data Visualization

Includes:

Bar charts for categorical variable distribution

Correlation heatmap

Key Insight:
Credit_History has the strongest positive correlation with Loan_Status.

5. Train-Test Split

80% Training

20% Testing

Target variable: Loan_Status

6. Model Training

Algorithms used:

Logistic Regression

Decision Tree

Random Forest

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Gradient Boosting

Training Accuracy:

Decision Tree — 100% (overfitting)

Random Forest — 96.03%

Gradient Boosting — 88.91%

Logistic Regression / KNN — ~80%

SVC — 68.41%

7. Model Evaluation (Testing)
Model	Accuracy
Logistic Regression	85%
Gradient Boosting	82.5%
Random Forest	80%
SVC	72.5%
Decision Tree	67.5%
KNN	60%

Best Performing Models:
➡️ Logistic Regression
➡️ Gradient Boosting
➡️ Random Forest

These showed good generalization on unseen data.

##Project Structure
├── data/
│   └── loan.csv
├── notebooks/
│   └── loan_prediction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model_training.py
│   └── evaluation.py
├── README.md
└── requirements.txt

##How to Run
1. Clone the Repository
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction

2. Install Dependencies
pip install -r requirements.txt

3. Run the Notebook
jupyter notebook


Open loan_prediction.ipynb and run all cells.

##Future Improvements

Hyperparameter tuning

Feature engineering

Use of advanced models like XGBoost, LightGBM

Deployment using Flask / Streamlit

API-based prediction service

##Author

Jeswin Thomas
Intern at Enterprise Building Training Solutions (EBTS)
Machine Learning & Data Science Enthusiast
