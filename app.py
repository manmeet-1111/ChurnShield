
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import streamlit as st



# Load trained model package 
model_bundle = joblib.load("churn_model.pkl")
ml_model = model_bundle['model']
encoder = model_bundle['preprocessor']
cut_point = model_bundle['threshold']



# Streamlit setup
st.set_page_config(page_title="ChurnShield", page_icon="📈", layout="wide")

st.title("📊 ChurnShield:Customer Churn Prediction System")
st.caption("Predicts whether a customer will discontinue the service or stay subscribed.")


# Background style
# Theme-aware styling for ChurnShield 
st.markdown("""
    <style>
    /* Make app background follow Streamlit's theme */
    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
    }

    /* Make sidebar and header transparent */
    [data-testid="stHeader"], [data-testid="stSidebar"] {
        background: transparent;
    }

    /* Keep buttons clearly visible in all modes */
    .stButton > button {
        border-radius: 8px;
        color: var(--text-color);
        background-color: var(--secondary-background-color);
        border: 1px solid var(--text-color);
    }
    .stButton > button:hover {
        background-color: var(--primary-color);
        color: white;
    }

    /* Adjust text colors dynamically */
    h1, h2, h3, p, label {
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)



# SINGLE CUSTOMER PREDICTION

st.header("🎯 Predict for One Customer")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet = st.selectbox("Internet Type", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col2:
        device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_help = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        stream_movie = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paper_bill = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0)
        total = st.number_input("Total Charges ($)", 10.0, 9000.0, 1000.0)

    submit_btn = st.form_submit_button("🔍 Predict")



# 🔹 PREDICTION LOGIC

if submit_btn:
    # --- Prepare Data ---
    form_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": ["No"],
        "InternetService": [internet],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protect],
        "TechSupport": [tech_help],
        "StreamingTV": [stream_tv],
        "StreamingMovies": [stream_movie],
        "Contract": [contract],
        "PaperlessBilling": [paper_bill],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
    })

    # Feature Engineering 
    form_data['TenureBucket'] = pd.cut(
        form_data['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )
    form_data['AvgBill'] = form_data['TotalCharges'] / (form_data['tenure'] + 1)
    form_data['LongTerm'] = (form_data['tenure'] >= 48).astype(int)
    form_data['ServiceCount'] = (
        (form_data['OnlineSecurity'] == 'Yes').astype(int) +
        (form_data['OnlineBackup'] == 'Yes').astype(int) +
        (form_data['DeviceProtection'] == 'Yes').astype(int) +
        (form_data['TechSupport'] == 'Yes').astype(int) +
        (form_data['StreamingTV'] == 'Yes').astype(int) +
        (form_data['StreamingMovies'] == 'Yes').astype(int)
    )
    form_data['NumAddOns'] = form_data[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                        'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
                                            lambda r: sum(v == 'Yes' for v in r), axis=1)
    form_data['AvgMonthlyCharges'] = form_data['TotalCharges'] / (form_data['tenure'] + 1)
    form_data['LongTenure'] = (form_data['tenure'] >= 48).astype(int)



    # Prediction
    X_new = encoder.transform(form_data)
    prob = ml_model.predict_proba(X_new)[:, 1][0]
    result = int(prob >= cut_point)

    # Show Results 
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        if result == 1:
            st.error(f"❌ Customer is **likely to leave**.\n\nChance of leaving: **{prob:.2%}**")
        else:
            st.success(f"✅ Customer is **likely to stay**.\n\nChance of leaving: **{prob:.2%}**")

    #Gauge Chart with Risk Colors 
    with col2:
        risk_color = "green" if prob < 0.4 else "yellow" if prob < 0.7 else "red"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={'text': "Churn Probability (%)"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 40], 'color': '#3cb371'},
                    {'range': [40, 70], 'color': '#ffeb3b'},
                    {'range': [70, 100], 'color': '#f44336'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

    # --- Smart Retention Tip ---
    st.markdown("---")
    if prob < 0.4:
        st.success("🟢 Low churn risk — customer is likely to stay.")
        st.info("💡 Tip: Keep engaging this customer with loyalty rewards or new service upgrades.")
    elif prob < 0.7:
        st.warning("🟡 Medium churn risk — monitor this customer closely.")
        st.info("💡 Tip: Offer limited-time discounts or personalized plans to strengthen loyalty.")
    else:
        st.error("🔴 High churn risk — customer likely to leave soon.")
        st.info("💡 Tip: Contact this customer directly and offer retention incentives such as a 20% discount or better service plan.")

    # --- Save prediction to CSV log ---
    result_text = "Leave" if result == 1 else "Stay"
    log_data = form_data.copy()
    log_data["Prediction"] = result_text
    log_data["Churn_Probability (%)"] = round(prob * 100, 2)

    # --- Safe logging folder ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "prediction_logs.csv")

    try:
        if not os.path.exists(log_path):
            log_data.to_csv(log_path, index=False)
        else:
            log_data.to_csv(log_path, mode='a', index=False, header=False)
        st.success("Prediction saved to 'logs/prediction_logs.csv' ✅")
    except PermissionError:
        st.warning("⚠️ Could not save log — please close 'prediction_logs.csv' if it's open in Excel and try again.")


# 📂 BULK PREDICTION SECTION

st.markdown("---")
st.header("📂 Bulk Customer Prediction (Upload CSV)")

uploaded = st.file_uploader("Upload a CSV file for multiple customers", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded data:", df.head())

  # --- Feature Engineering for CSV data ---
    df['TenureBucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )
    df['AvgBill'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['LongTerm'] = (df['tenure'] >= 48).astype(int)
    df['ServiceCount'] = (
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int)
    )

    #  Add engineered columns that the model expects
    df['LongTenure'] = (df['tenure'] >= 48).astype(int)
    df['NumAddOns'] = df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
                            lambda r: sum(v == 'Yes' for v in r), axis=1)
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)

    # --- Predictions ---
    processed = encoder.transform(df)
    probs = ml_model.predict_proba(processed)[:, 1]
    df['Churn_Probability (%)'] = (probs * 100).round(2)
    df['Prediction'] = ['Leave' if p >= cut_point else 'Stay' for p in probs]

    st.subheader("Predictions:")
    st.dataframe(df[['gender', 'Contract', 'MonthlyCharges', 'Churn_Probability (%)', 'Prediction']].head())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")
