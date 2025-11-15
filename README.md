Loan Approval Prediction using Machine Learning

This project focuses on predicting loan approval status using machine learning models trained on applicant demographic, financial, and credit-related data. The goal is to assist financial institutions in faster and more accurate decision-making by identifying whether a loan application is likely to be approved.

Project Overview

The project follows a complete machine-learning workflow:

Import and inspect dataset

Exploratory data analysis (EDA)

Data preprocessing and handling missing values

Label encoding of categorical features

Data visualization

Model training using various ML algorithms

Model evaluation on unseen data

The work was completed as part of an internship project under Enterprise Building Training Solutions (EBTS).

Organization Overview

Enterprise Building Training Solutions (EBTS) is a nationwide training and project-execution organization specializing in Data Science, Machine Learning, and Software Engineering.
The program focuses on hands-on, mentor-guided learning emphasizing clear communication, documentation, and reproducible project execution.

Internship guided by Mr. Arunjit Chowdhury, CEO of EBTS.

Problem Statement

Build a predictive system to classify whether a loan application will be Approved (Y) or Not Approved (N) using historical loan application data.

Tech Stack & Tools
Hardware

Intel Core i7-12650H @ 2.30 GHz

16 GB RAM

GPU: 6 GB

Software & Libraries

Python 3.8+

pandas, NumPy – Data processing

matplotlib, seaborn – Visualization

scikit-learn – Machine Learning

Jupyter Notebook – Development environment

Project Workflow
1. Import Libraries & Load Dataset

Dataset loaded using pandas

Schema validation & type checking performed

Initial exploratory analysis conducted

2. Data Preprocessing

Dropped unnecessary columns (e.g., Loan_ID)

Handled missing values using column mean

Ensured data integrity before encoding

3. Data Encoding

Label encoding applied to convert categorical fields to numeric values

Ensured the final dataset contains only numeric features

4. Data Visualization

Visualizations include:

Bar charts for categorical variable distribution

Correlation heatmap to study feature relationships

Key observation:
 Credit_History has the strongest positive correlation with Loan_Status.

5. Train-Test Split

80% training

20% testing

Loan_Status used as the target variable

6. Model Training

Models used:

Logistic Regression

Decision Tree

Random Forest

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Gradient Boosting

Training accuracies showed:

Decision Tree: 100% (overfitting)

Random Forest: 96.03%

Gradient Boosting: 88.91%

Logistic Regression & KNN: ~80%

SVC: 68.41%

7. Model Evaluation

Testing accuracy:

Model	Accuracy
Logistic Regression	85%
Gradient Boosting	82.5%
Random Forest	80%
SVC	72.5%
Decision Tree	67.5%
KNN	60%

Best performing models:
Logistic Regression, Gradient Boosting, Random Forest (good generalization)

Project Structure (recommended)
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

How to Run

Clone this repository:

git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook


Open loan_prediction.ipynb and execute all cells.

Future Improvements

Hyperparameter tuning

Feature engineering

Using XGBoost or LightGBM

Building a Flask/Streamlit UI

Deploying the model via API

Author

Jeswin Thomas
Intern at Enterprise Building Training Solutions (EBTS)
Machine Learning & Data Science Enthusiast
