import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Page config
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Safe model load with caching
@st.cache_resource
def load_model():
    model_path = "fraud_detection_pipeline.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Model load error: {str(e)}")
            return None
    else:
        st.error("❌ Model file 'fraud_detection_pipeline.pkl' not found! Keep it in the same folder.")
        st.info("Current files: " + ", ".join(os.listdir(".")))
        return None

# Load model
model = load_model()
if model is None:
    st.stop()

st.title("🔍 Fraud Detection Prediction App")
st.markdown("### Enter transaction details and click Predict")

st.divider()

# Input form in columns for better UI
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (Transaction time step)", min_value=0, value=1, step=1)
    transaction_type = st.selectbox("Transaction Type", 
                                   ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "CASH_IN"])
    amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, step=100.0)

with col2:
    oldbalanceOrg = st.number_input("Old Balance Sender ($)", min_value=0.0, value=10000.0)
    newbalanceOrig = st.number_input("New Balance Sender ($)", min_value=0.0, value=9000.0)
    oldbalanceDest = st.number_input("Old Balance Receiver ($)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance Receiver ($)", min_value=0.0, value=1000.0)

# Predict button
if st.button("🚀 Predict Fraud", type="primary"):
    # Create input dataframe (standard PaySim columns)
    input_data = pd.DataFrame({
        "step": [step],
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })
    
    st.write("**Input Data:**")
    st.dataframe(input_data, use_container_width=True)
    
    try:
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
        
        # Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", "FRAUD" if prediction == 1 else "SAFE", 
                     delta="High Risk" if prediction == 1 else "Low Risk")
        
        if probability is not None:
            with col2:
                st.metric("Fraud Probability", f"{probability[1]*100:.1f}%")
            with col3:
                st.metric("Safe Probability", f"{probability[0]*100:.1f}%")
        
        # Alert
        if prediction == 1:
            st.error("🚨 **ALERT: This transaction may be FRAUD!** Please investigate.")
        else:
            st.success("✅ **SAFE: Transaction appears normal.**")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("Check if model expected columns match training data.")

