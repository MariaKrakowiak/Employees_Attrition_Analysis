# Attrition Analysis Project

This project is a Streamlit web application for visualizing and analyzing employee attrition using the IBM HR Analytics dataset.

## Features

- Data exploration (distributions, relationships)
- Model evaluation (Logistic Regression, Random Forest, Decision Tree, Ada Boost, Gradient Boosting, XGBoost, Catboost, SMOTE, sampling strategies)
- Confusion matrices, ROC curves and performance metrics
- User-friendly interface via Streamlit [Stay-or-go-app](https://stay-or-go.streamlit.app/)

## Dataset

Dataset used: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Installation & Local Run Instructions

To run the application locally:

1. Go to the GitHub repository and download the ZIP of the project.
2. Unzip the folder to your preferred location.
3. Open a terminal or command prompt.
4. Navigate to the folder where the project was extracted:

   ```bash
   cd path_to_folder
5. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
6. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

7. Launch the Streamlit app:
   ```bash
   streamlit run attrition_app.py

The app will open in your default web browser at http://localhost:8501.