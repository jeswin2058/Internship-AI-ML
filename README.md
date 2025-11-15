# Loan Approval Prediction using Machine Learning

This project focuses on predicting loan approval status using machine learning models trained on applicant demographic, financial, and credit-related data. The objective is to assist financial institutions in faster and more accurate decision-making by identifying whether a loan application is likely to be approved.

## Project Overview
This project includes a complete end-to-end machine learning workflow:
- Importing and inspecting the dataset
- Exploratory Data Analysis (EDA)
- Data preprocessing and handling missing values
- Encoding categorical variables
- Data visualization
- Model training using various machine learning algorithms
- Model evaluation on unseen data

This work was completed as part of an internship project under Enterprise Building Training Solutions (EBTS).

## Organization Overview
Enterprise Building Training Solutions (EBTS) is a nationwide training and project-execution organization specializing in Data Science, Machine Learning, and Software Engineering. Its programs emphasize mentor-guided, hands-on learning with a focus on clear documentation and reproducibility.

This internship was conducted under the mentorship of Mr. Arunjit Chowdhury, CEO of EBTS.

## Problem Statement
To build a predictive system that classifies whether a loan application will be:
- Approved (Y), or
- Not Approved (N)

based on historical loan application data.

## Tech Stack and Tools

### Hardware
- Intel Core i7-12650H @ 2.30 GHz
- 16 GB RAM
- GPU: 6 GB

### Software and Libraries
- Python 3.8+
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

## Project Workflow

### 1. Import Libraries & Load Dataset
- Dataset loaded using pandas
- Schema validation and datatype checks performed
- Initial exploratory analysis completed

### 2. Data Preprocessing
- Removed unnecessary columns (e.g., Loan_ID)
- Filled missing values using column means
- Ensured consistent formatting across features

### 3. Data Encoding
- Applied Label Encoding to convert categorical fields to numeric values
- Ensured the dataset contained only numerical features

### 4. Data Visualization
Generated visualizations including:
- Bar charts for categorical variable distributions
- Correlation heatmap

**Key finding:** Credit_History shows the strongest positive correlation with Loan_Status.

### 5. Train-Test Split
- 80% training data
- 20% testing data
- Target variable: Loan_Status

### 6. Model Training
Models used:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

Training accuracy summary:
- Decision Tree: 100% (overfitting observed)
- Random Forest: 96.03%
- Gradient Boosting: 88.91%
- Logistic Regression & KNN: ~80%
- SVC: 68.41%

### 7. Model Evaluation (Testing)

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 85%      |
| Gradient Boosting    | 82.5%    |
| Random Forest        | 80%      |
| SVC                  | 72.5%    |
| Decision Tree        | 67.5%    |
| KNN                  | 60%      |

**Best performing models:**
- Logistic Regression  
- Gradient Boosting  
- Random Forest  

These demonstrated the best generalization on unseen test data.

## Recommended Project Structure
```
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
```

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook
```

Open `loan_prediction.ipynb` and run all cells.

## Future Improvements
- Hyperparameter tuning
- Feature engineering
- Use of XGBoost or LightGBM
- Building a Flask or Streamlit web interface
- Deploying the model through an API

## Author
**Jeswin Thomas**  
Intern at Enterprise Building Training Solutions (EBTS)  
Machine Learning & Data Science Enthusiast
