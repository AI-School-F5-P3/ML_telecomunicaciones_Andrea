import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
)

# Load Models
@st.cache_resource
def load_model(model_name, directory='models'):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory, f"{model_name}.pkl")
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data['model'], model_data['preprocessor']

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
    # Load the preprocessor and the Random Forest model (done once for efficiency)
    preprocessor = joblib.load('models/preprocessor.pkl')
    rf_model = joblib.load('models/Random_Forest_For_Streamlit.pkl')

    # Define the prediction function
    def predict_customer_category(data):
        processed_data = preprocessor.transform(data)
        prediction = rf_model.predict(processed_data)
        return prediction

    # Streamlit UI
    st.title("Customer Category Prediction")
    st.write("Use this app to predict customer categories based on input features.")

    # Input form
    with st.form("prediction_form"):
        tenure = st.number_input("Tenure", min_value=0)
        age = st.number_input("Age", min_value=0)
        address = st.number_input("Address Years", min_value=0)
        income = st.number_input("Income", min_value=0)
        employ = st.number_input("Years Employed", min_value=0)
        reside = st.number_input("Years Resided", min_value=0)
        region = st.selectbox("Region", ['1', '2', '3', '4'])
        marital = st.selectbox("Marital Status", ['Single', 'Married'])
        ed = st.selectbox("Education Level", ['1', '2', '3', '4'])
        retire = st.selectbox("Retire", ['0', '1'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Create DataFrame with input values
            input_data = pd.DataFrame({
                'tenure': [tenure],
                'age': [age],
                'address': [address],
                'income': [income],
                'employ': [employ],
                'reside': [reside],
                'region': [region],
                'marital': [marital],
                'ed': [ed],
                'retire': [retire],
                'gender': [gender]
            })
            
            # Perform prediction
            prediction = predict_customer_category(input_data)
            st.write("Predicted Customer Category:", prediction[0])

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