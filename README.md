# Financial-Fraud-Detection-ML

An end-to-end Machine Learning solution designed to identify fraudulent financial transactions. This project leverages a large-scale dataset to train a classification model and features an interactive web dashboard for real-time risk assessment.

🚀 Key Features
High Accuracy: Achieved a 94% prediction accuracy using a Logistic Regression model.

Large Scale Data Analysis: Processed and analyzed a 6.3 Million records Kaggle dataset (PaySim).

Real-time Prediction: Built an interactive web application using Streamlit to assess transaction risk instantly.

Optimized ML Pipeline: Implemented Scikit-learn Pipelines and ColumnTransformers for automated data scaling and categorical encoding.

Imbalance Handling: Utilized class_weight='balanced' and stratified splitting to handle highly skewed fraud data.

🛠️ Tech Stack & Environment
Language: Python

Environment: VS Code (Jupyter Extension)

ML Libraries: Scikit-learn, Pandas, NumPy

Visualization: Seaborn, Matplotlib

Deployment: Streamlit, Joblib

📊 Project Workflow
Exploratory Data Analysis (EDA): Performed statistical profiling and visualized fraud patterns across transaction types (TRANSFER, CASH_OUT).

Feature Engineering: Handled data cleaning, log transformations for transaction amounts, and feature encoding.

Model Training: Developed a classification pipeline to automate preprocessing and model fitting.

Evaluation: Validated the model using Confusion Matrix and Classification Reports (Precision, Recall, F1-Score).

Deployment: Exported the pipeline using Joblib and integrated it into a Streamlit dashboard.

:

📊 Dataset Information
The dataset used for this project is the PaySim synthetic dataset from Kaggle, which simulates mobile money transactions.

Total Records: 6.3 Million+ transactions.

Download Link : https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset
