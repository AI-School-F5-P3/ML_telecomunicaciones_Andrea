import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
)

# Load Models
@st.cache_resource
def load_model():
    model_path = "src/models/Random_Forest.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Home Page
def home_page():
    st.title("Customer Segmentation Project")
    st.subheader("Planteamiento del Problema")
    st.write("""
    Una empresa de telecomunicaciones ha segmentado a sus clientes en cuatro grupos
    basados en sus patrones de uso del servicio. El objetivo es desarrollar un modelo de
    clasificación que, utilizando datos demográficos como región, edad, ingresos, etc., prediga
    el grupo al que pertenece un cliente. Este modelo permitirá personalizar las ofertas y
    servicios para clientes prospectivos, mejorando la efectividad de las campañas de
    marketing.
    """)
    
    st.subheader("Impacto Esperado")
    st.write("""
    El proyecto busca desarrollar un sistema que facilite la segmentación de clientes basándose
    en datos demográficos, permitiendo una personalización efectiva de ofertas y servicios. Esto
    no solo mejorará la experiencia del cliente, sino que también optimizará los recursos y
    estrategias de marketing de la empresa.
    """)

# Metrics Page
def metrics_page():
    st.title("Model Metrics")
    st.write("Use the dropdown menu to view the metrics of each model.")
    
    # Dropdown for selecting the model
    model_options = {
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "K-Nearest Neighbors": "k_nearest_neighbors"
    }
    selected_model = st.selectbox("Select Model:", list(model_options.keys()))
    
    # Load the appropriate model
    model_name = model_options[selected_model]
    st.write(f"Metrics for **{selected_model}**:")
    
    # Load pre-computed metrics (or display example)
    # Example metrics are placeholders. Replace with real model evaluation results.
    accuracy = np.random.uniform(0.7, 0.9)  # Example metric
    precision = np.random.uniform(0.6, 0.8)
    recall = np.random.uniform(0.6, 0.8)
    f1_score = np.random.uniform(0.6, 0.8)
    
    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("Precision", f"{precision:.4f}")
    st.metric("Recall", f"{recall:.4f}")
    st.metric("F1-Score", f"{f1_score:.4f}")
    
    st.write("### Confusion Matrix:")
    st.write(confusion_matrix([1, 0, 1, 2, 3], [1, 0, 1, 2, 2]))  # Replace with real confusion matrix

    st.write("### Classification Report:")
    st.text(classification_report([1, 0, 1, 2, 3], [1, 0, 1, 2, 2]))  # Replace with real classification report

# Prediction Page
def prediction_page():
    st.title("Make Predictions")
    st.write("Use the form below to input customer data and predict their segment.")
    
    # Input fields for user data
    tenure = st.number_input("Tenure (years):", min_value=0, max_value=50, step=1)
    age = st.number_input("Age:", min_value=18, max_value=100, step=1)
    address = st.number_input("Address (years at residence):", min_value=0, max_value=50, step=1)
    income = st.number_input("Income (annual):", min_value=0, step=1000)
    employ = st.number_input("Employment (years):", min_value=0, max_value=50, step=1)
    reside = st.number_input("Reside (years in current area):", min_value=0, max_value=50, step=1)
    region = st.selectbox("Region:", [0, 1, 2, 3])  # Example values for one-hot encoded regions
    marital = st.selectbox("Marital Status:", [0, 1])  # Example values for one-hot encoded marital status
    ed = st.selectbox("Education Level:", [0, 1, 2, 3])  # Example values for one-hot encoded education levels
    retire = st.selectbox("Retire Status:", [0, 1])
    gender = st.selectbox("Gender:", [0, 1])  # Example values for one-hot encoded gender
    
    # Predict button
    if st.button("Predict"):
        # Prepare the input data
        input_data = np.array([[tenure, age, address, income, employ, reside, region, marital, ed, retire, gender]])
        
        # Load the best model
        best_model = load_model()
        
        # Make prediction
        prediction = best_model.predict(input_data)
        
        # Display prediction
        st.write(f"Predicted Customer Segment: **{prediction[0]}**")

# App Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Model Metrics", "Make Predictions"]
)

if page == "Home":
    home_page()
elif page == "Model Metrics":
    metrics_page()
elif page == "Make Predictions":
    prediction_page()