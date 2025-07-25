import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ Constants ------------------ #
CATEGORICAL_COLUMNS = [
    'BusinessTravel', 'Department', 'EducationField',
    'JobRole', 'MaritalStatus', 'OverTime'
]

NUMERICAL_COLUMNS = [
    'YearsAtCompany', 'JobSatisfaction', 'EnvironmentSatisfaction', 'DistanceFromHome',
    'YearsWithCurrManager', 'YearsInCurrentRole', 'NumCompaniesWorked', 'TotalWorkingYears',
    'YearsSinceLastPromotion'
]

CATEGORICAL_OPTIONS = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                'Manufacturing Director', 'Healthcare Representative', 'Manager',
                'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

CATEGORICAL_ENCODINGS = {
    'OverTime': {'Yes': 1, 'No': 0}
}

DISPLAY_NAMES = {
    'BusinessTravel': 'Business Travel frequency of employee',
    'Department': 'Department of employee',
    'EducationField': 'Education Field of employee',
    'JobRole': 'Job Role of employee',
    'MaritalStatus': 'Marital Status of employee',
    'OverTime': 'OverTime for employee',
    'DistanceFromHome': 'Distance From Home for employee',
    'EnvironmentSatisfaction': 'Environment Satisfaction for employee',
    'JobSatisfaction': 'Job Satisfaction for employee',
    'NumCompaniesWorked': 'Number of Companies Worked for employee',
    'TotalWorkingYears': 'Total Working Years for employee',
    'YearsAtCompany': 'Years at Company for employee',
    'YearsInCurrentRole': 'Years in Current Role for employee',
    'YearsSinceLastPromotion': 'Years Since Last Promotion for employee',
    'YearsWithCurrManager': 'Years with Current Manager for employee'
}


# ------------------ Model Loading ------------------ #
def load_model(relative_path: str):
    base_dir = os.path.dirname(__file__)  # directory of current script
    model_path = os.path.join(base_dir, relative_path)
    with open(model_path, 'rb') as file:
        return pickle.load(file)


# ------------------ Page: Overview ------------------ #
def page_overview():
    st.markdown("<h1 style='text-align: center;'>Stay or Go App</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> Welcome to the Stay or Go Application!</h2>", unsafe_allow_html=True)
    st.markdown("""
       <div style='text-align: justify; font-size: 18px'>
       Understanding why employees leave is crucial for building a healthy, productive and stable workplace. This application is designed to help organizations predict employee attrition
       using data-driven insights.<br><br>

       <strong style='font-size: 24px'>Why it matters</strong>

       - <strong>REDUCE TURNOVER COSTS</strong> - Losing skilled employees is expensive. Predicting attrition early allows companies to take action before it happens.
       - <strong>DATA-DRIVEN HR DECISIONS</strong> - Moves HR strategy from reactive to proactive, with decisions backed by machine learning.
       - <strong>IMPROVE EMPLOYEE SATISFACTION</strong> - Identify dissatisfaction factors and implement meaningful improvements in working conditions or policies.
       - <strong>TARGETED RETENTION STRATEGIES</strong> - Focus resources on employees most at risk of leaving — where interventions will matter most.
       - <strong>SUPPORT WORKFORCE PLANNING</strong> - Enable better succession planning, hiring forecasts and training investments.

       <hr style='border: 2px solid #ccc;'>
       <strong style='font-size: 24px'>Why I built this project</strong><br>

       I created this project to better understand the factors that drive people to leave their jobs. 

       It was also a valuable opportunity to refresh and strengthen my machine learning skills by working through the entire modeling pipeline — from data preprocessing to model evaluation. 

       I chose the [IBM HR Analytics Attrition Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) because it presents a rich and realistic business scenario with multiple variables influencing employee behavior.

       While the original dataset contains a wide range of features, I focused on selecting only those that had the most significant impact on model performance. This approach helped simplify the model while retaining predictive power and interpretability. It also makes the final application user friendly.

       The whole analysis and source code are available here: [GitHub Repository – Employees Attrition Analysis](https://github.com/MariaKrakowiak/Employees_Attrition_Analysis).

    </div>
    

       """, unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
    st.markdown(
        """
        ### 👈 Use the tabs on the left to explore data or predict if employee is likely to leave or stay in the company.
        """
    )
    img_path = os.path.join(os.path.dirname(__file__), "..", "images", "people.jpg")
    st.image(img_path, use_container_width=True)


# ------------------ Page: Data Plot ------------------ #
def page_plot(df):
    st.markdown("<h1 style='text-align: center;'>Dataset exploration and visualization</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style = 'text-align: center; font-size: 18px'>Here is a quick overview of the employee dataset:</div><br>
    """, unsafe_allow_html=True)
    st.dataframe(df.head())

    col3, col4, col5, col6 = st.columns(4)

    col3.metric(label="Total rows", value=len(df))
    col4.metric(label="Total columns", value=len(df.columns))
    col5.metric(label="Target variable", value="Attrition")
    col6.metric(label="Kind of problem", value="two-label task")

    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
    features = {
        "Demographics ": "Age, Gender, MaritalStatus, Education, EducationField",
        "Job-Related": "JobRole, Department, JobSatisfaction, JobLevel, MonthlyIncome",
        "Performance & Environment": "PerformanceRating, WorkLifeBalance, EnvironmentSatisfaction",
        "Work Conditions": "OverTime, DistanceFromHome, YearsAtCompany, NumCompaniesWorked"
    }

    cols1 = st.columns(2)
    cols2 = st.columns(2)

    for idx, (category, variables) in enumerate(features.items()):
        card_html = f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #f1f3f6; 
                        box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin: 10px 0;'>
                <h4 style='margin-bottom: 0.5rem;'>{category}</h4>
                <p style='margin: 0; font-size: 14px; color: #444;'>{variables}</p>
            </div>
        """
        if idx < 2:
            cols1[idx].markdown(card_html, unsafe_allow_html=True)
        else:
            cols2[idx - 2].markdown(card_html, unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    st.markdown("""
        <div style = 'text-align: justify; font-size: 18px'>Below charts present the features: <strong>OverTime</strong> and <strong>JobSatisfaction</strong> which have the strongest correlation with <strong>Attrition</strong>.</div><br>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### OverTime vs Attrition")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='OverTime', hue='Attrition', palette='Set2', ax=ax)
        ax.set_xlabel("OverTime")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.markdown("""
        <div style = 'font-size: 18px'>
        <ul>
        <li>A much higher proportion of employees who work overtime have left the company compared to those we do not.</li>
        <li>Overtime is often associated with stress, burnout and poor work-life balance, all of which are strong drivers of attrition.</li>
        <li>This makes OverTime one of the strongest behavioral indicators of an employee's likelihood to quit.</li?
        </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Job Satisfaction vs Attrition")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, palette='Set2', ax=ax2)
        ax2.set_xlabel("JobSatisfaction")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        st.markdown("""
                <div style = 'font-size: 18px'>
                <ul>
                <li>Attrition is higher among employees with lower job satisfaction.</li>
                <li>Employees reporting low satisfaction are more likely to seek new opportunities.</li>
                <li>This confirms that subjective experiences like satisfaction directly relate to retention.</li>
                </ul>
                </div>""",
                    unsafe_allow_html=True)


# ------------------ Page: Prediction ------------------ #
def get_user_input():
    user_input = {}

    col1, col2 = st.columns(2)

    # NUMERICAL FEATURES
    with col1:
        st.markdown("<h4 style='font-size: 22px;'>Numerical Features</h4>", unsafe_allow_html=True)
        for col in NUMERICAL_COLUMNS:
            label = DISPLAY_NAMES.get(col, col)
            styled_label = f"<span style='font-size:18px; margin-bottom:-8px'>{label}</span>"
            st.markdown(styled_label, unsafe_allow_html=True)

            if col == 'Age':
                user_input[col] = st.slider("", min_value=18, max_value=60, value=30, key=f"{col}_slider",
                                            label_visibility="collapsed")
            elif col in ['MonthlyIncome', 'MonthlyRate']:
                user_input[col] = st.slider("", min_value=1000, max_value=20000, step=500, value=5000,
                                            key=f"{col}_slider", label_visibility="collapsed")
            else:
                user_input[col] = st.number_input("", step=1, min_value=0, format="%d", key=f"{col}_input",
                                                  label_visibility="collapsed")

    # CATEGORICAL FEATURES
    with col2:
        st.markdown("<h4 style='font-size: 22px;'>Categorical Features</h4>", unsafe_allow_html=True)
        for col in CATEGORICAL_COLUMNS:
            label = DISPLAY_NAMES.get(col, col)
            options = CATEGORICAL_OPTIONS.get(col, [])
            styled_label = f"<span style='font-size:18px; margin-bottom:-8px'>{label}</span>"
            st.markdown(styled_label, unsafe_allow_html=True)

            if col in ['BusinessTravel', 'Department', 'EducationField', 'JobRole']:
                value = st.selectbox("", options, key=f"{col}_select", label_visibility="collapsed")
            else:
                value = st.radio("", options, horizontal=True, key=f"{col}_radio", label_visibility="collapsed")

            user_input[col] = CATEGORICAL_ENCODINGS.get(col, {}).get(value, value)

    return user_input


def create_input_dataframe(user_input: dict, expected_columns: list):
    input_df = pd.DataFrame([user_input])
    for col in NUMERICAL_COLUMNS:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    for col in CATEGORICAL_COLUMNS:
        input_df[col] = input_df[col].astype(str)
    input_df = input_df[expected_columns]
    return input_df


def display_prediction(prediction):
    if prediction == 1:
        st.error("The employee is likely to **leave** the company.")
        st.snow()
    else:
        st.success("The employee is likely to **stay** at the company.")
        st.balloons()


def page_predict(model):
    st.markdown("<h1 style='text-align: center;'>Predict if the employee is likely to stay or leave</h1>",
                unsafe_allow_html=True)
    st.markdown("""
        <div style = 'text-align: center; font-size: 18px'>Fill the form to check if the employee will leave or stay:</div><br>""",
                unsafe_allow_html=True)

    user_input = get_user_input()
    if st.button("Predict", help="Click to predict based on current input"):
        input_df = create_input_dataframe(user_input, model.feature_names_in_)
        if input_df.isnull().values.any():
            st.error("Please ensure all fields are correctly filled.")
            st.write(input_df)
        else:
            prediction = make_prediction(model, input_df)
            display_prediction(prediction)


def make_prediction(model, input_df):
    return model.predict(input_df)[0]

# ------------------ Main ------------------ #
def main():
    st.set_page_config(page_title="Stay or Go App", layout="wide")
    st.markdown("<style>body { background-color: #f7f7f7; }</style>", unsafe_allow_html=True)

    page = st.sidebar.selectbox("Select Page", ["Overview", "Explore Data", "Predict"])

    model = load_model("../model/trained_model.sav")

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.DataFrame()

    if page == "Overview":
        page_overview()
    elif page == "Explore Data":
        if not df.empty:
            page_plot(df)
        else:
            st.warning("No dataset available for visualization.")
    elif page == "Predict":
        page_predict(model)


if __name__ == "__main__":
    main()
