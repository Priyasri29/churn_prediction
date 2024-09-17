import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from src.data_utils import load_data, preprocess_data
from src.model import train_model, save_model, load_model

# Set up the page
st.set_page_config(page_title="Flexible Churn Prediction App", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Loading", "Prediction", "Explainable AI"])

# Data Loading and Preprocessing Page
if page == "Data Loading":
    st.title("Customer Churn Prediction - Data Loading")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])

    if uploaded_file:
        # Load and display raw data
        data = load_data(uploaded_file)
        st.write("### Raw Data Preview")
        st.dataframe(data.head())

        # Preprocess data
        preprocessed_data, label_encoders = preprocess_data(data)
        st.write("### Preprocessed Data Preview")
        st.dataframe(preprocessed_data.head())

        # Store the processed data in session state for future use
        st.session_state['preprocessed_data'] = preprocessed_data
        st.session_state['label_encoders'] = label_encoders
    else:
        st.warning("Please upload a dataset to proceed.")

# Prediction Page with Dynamic Form Structure
elif page == "Prediction":
    st.title("Model Training and Prediction")

    if 'preprocessed_data' in st.session_state:
        preprocessed_data = st.session_state['preprocessed_data']

        # Split the dataset into features (X) and target (y)
        X = preprocessed_data.drop('churn', axis=1)
        y = preprocessed_data['churn']

        # Train the model with cross-validation and regularization
        if st.button("Train Model"):
            model = RandomForestClassifier()
            param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X, y)
            model = grid_search.best_estimator_

            save_model(model, "model.pkl")
            st.success(f"Model trained and saved successfully with params: {grid_search.best_params_}!")

        # Predict on new data based on dataset structure
        if st.button("Predict"):
            model = load_model("model.pkl")

            st.write("### Enter new data for prediction")
            user_input = {}
            for col in X.columns:
                user_input[col] = st.text_input(f"Input {col}")

            if st.button("Get Prediction"):
                input_data = np.array([list(user_input.values())])
                prediction = model.predict(input_data)

                st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
                st.write("Note: This prediction is based on the model trained with the uploaded data.")
    else:
        st.warning("Please upload and preprocess data in the 'Data Loading' page first.")

# Explainable AI Page
elif page == "Explainable AI":
    st.title("Explain Model Predictions with Explainable AI")

    if 'preprocessed_data' in st.session_state:
        preprocessed_data = st.session_state['preprocessed_data']
        X = preprocessed_data.drop('churn', axis=1)

        # Load the trained model
        model = load_model("model.pkl")

        # Option to select between SHAP and LIME
        explainer_choice = st.selectbox("Choose explainer", ["SHAP", "LIME"])

        if explainer_choice == "SHAP":
            st.write("### SHAP Global Explanation")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            st.write("#### SHAP Summary Plot")
            shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')

            st.write("### SHAP Local Explanation")
            row = st.number_input("Choose a row number to explain:", min_value=0, max_value=len(X)-1, value=0)
            st.write("#### SHAP Force Plot")
            shap.force_plot(explainer.expected_value[1], shap_values[1][row,:], X.iloc[row,:], matplotlib=True)
            st.pyplot(bbox_inches='tight')

        elif explainer_choice == "LIME":
            st.write("### LIME Explanation")

            explainer = LimeTabularExplainer(X.values, mode='classification', feature_names=X.columns, class_names=['Not Churn', 'Churn'], verbose=True)

            row = st.number_input("Choose a row number to explain:", min_value=0, max_value=len(X)-1, value=0)

            explanation = explainer.explain_instance(X.iloc[row].values, model.predict_proba)
            st.write(explanation.as_list())
            explanation.show_in_notebook(show_table=True)
    else:
        st.warning("Please train the model first in the 'Prediction' page.")