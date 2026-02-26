import streamlit as st
import pandas as pd
import numpy as np
import base64
import sqlite3
from datetime import datetime


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Loan Prediction | Data Vidwan",
    page_icon="💰",
    layout="wide"
)

# ================= BACKGROUND COLOR =================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e3f2fd, #fce4ec);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #ffffff, #e1f5fe);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect("loan_predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    age INTEGER,
    annual_income REAL,
    credit_score INTEGER,
    loan_amount REAL,
    probability REAL,
    result TEXT
)
""")
conn.commit()

# ---------------- LOAD MODEL ----------------
import joblib
model = joblib.load("loan_xgboost_pipeline.pkl")

# ---------------- LOAD LOGO ----------------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("data_vidwan_logo.png")

# ================= SIDEBAR =================
st.sidebar.markdown(
    f"""
    <div style="text-align:center; padding:20px 0;">
        <img src="data:image/png;base64,{logo_base64}" width="240">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("📋 Customer Input Panel")

age = st.sidebar.slider("Age", 18, 70, 30)
years_employed = st.sidebar.number_input("Years Employed", 0.0, 50.0, 5.0, step=0.1)
annual_income = st.sidebar.number_input("Annual Income", 0, 100000000, 50000, step=1000)
credit_score = st.sidebar.slider("Credit Score", 100, 850, 650)
credit_history_years = st.sidebar.slider("Credit History Years", 0.1, 30.0, 5.0)

savings_assets = st.sidebar.number_input("Savings / Assets", 0, value=20000000, step=1000)
current_debt = st.sidebar.number_input("Current Debt", 0, value=10000000, step=1000)

defaults_on_file = st.sidebar.slider("Defaults on File", 0, 5, 0)
delinquencies_last_2yrs = st.sidebar.slider("Delinquencies (Last 2 Years)", 0, 10, 0)
derogatory_marks = st.sidebar.slider("Derogatory Marks", 0, 5, 0)

loan_amount = st.sidebar.number_input("Loan Amount", 0, 100000000, 50000, step=1000)
interest_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.0)

occupation_status = st.sidebar.selectbox(
    "Occupation Status",
    ["Employed", "Self-Employed", "Student", "Unemployed"]
)

product_type = st.sidebar.selectbox(
    "Product Type",
    ["Credit Card", "Personal Loan", "Line of Credit"]
)

loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    ["Business", "Home Improvement", "Medical", "Education", "Personal", "Debt Consolidation"]
)

debt_to_income_ratio = st.sidebar.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)
loan_to_income_ratio = st.sidebar.slider("Loan to Income Ratio", 0.0, 2.0, 0.5)
payment_to_income_ratio = st.sidebar.slider("Payment to Income Ratio", 0.0, 1.0, 0.2)

# ================= MAIN HEADER =================
st.markdown("""
<h1 style='text-align:center; color:#1a237e; font-weight:700;'>
🏦💰 AI-Driven Loan Approval Prediction System 🚀
</h1>
<hr>
""", unsafe_allow_html=True)

# ================= PREDICTION =================
st.subheader("📊 Prediction Result")

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if st.button("🔍 Predict Loan Status"):

    input_data = pd.DataFrame([{
        "customer_id": 1,
        "age": age,
        "occupation_status": occupation_status,
        "years_employed": years_employed,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "credit_history_years": credit_history_years,
        "savings_assets": savings_assets,
        "current_debt": current_debt,
        "defaults_on_file": defaults_on_file,
        "delinquencies_last_2yrs": delinquencies_last_2yrs,
        "derogatory_marks": derogatory_marks,
        "product_type": product_type,
        "loan_intent": loan_intent,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_to_income_ratio": loan_to_income_ratio,
        "payment_to_income_ratio": payment_to_income_ratio
    }])

    input_data["annual_income_log"] = np.log1p(annual_income)
    input_data["loan_amount_log"] = np.log1p(loan_amount)

    trained_cols = model.feature_names_in_
    for col in trained_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[trained_cols]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    probability_percent = round(probability * 100, 2)

    result_text = "Approved" if prediction == 1 else "Rejected"

    cursor.execute("""
    INSERT INTO predictions (timestamp, age, annual_income, credit_score, loan_amount, probability, result)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        age,
        annual_income,
        credit_score,
        loan_amount,
        probability_percent,
        result_text
    ))
    conn.commit()

    st.session_state.last_prediction = {
        "prediction": prediction,
        "probability": probability_percent,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate
    }

# ================= SHOW RESULT =================
if st.session_state.last_prediction:

    pred = st.session_state.last_prediction["prediction"]
    prob = st.session_state.last_prediction["probability"]
    loan_amt = st.session_state.last_prediction["loan_amount"]
    rate = st.session_state.last_prediction["interest_rate"]

    if pred == 1:

        st.success("🟢 LOAN APPROVED")

        st.markdown("""
        <div style="
            background-color:#e8f5e9;
            padding:20px;
            border-radius:12px;
            border-left:6px solid #2e7d32;
            font-size:18px;
            font-weight:500;
            color:#1b5e20;
        ">
            🎉 <b>Congratulations!</b><br><br>
            Your loan application has been successfully approved.
            Our system has evaluated your financial profile positively.
        </div>
        """, unsafe_allow_html=True)

        st.info(f"Approval Probability: {prob}%")

        tenure_years = 5
        months = tenure_years * 12
        monthly_rate = rate / 100 / 12

        emi = (loan_amt * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        total_payment = emi * months

        st.divider()
        st.subheader("📊 Financial Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Loan Amount", f"₹{loan_amt:,}")
        col2.metric("Monthly EMI", f"₹{round(emi,2):,}")
        col3.metric("Total Repayment", f"₹{round(total_payment,2):,}")

    else:

        st.error("🔴 LOAN REJECTED")
        st.warning(f"Approval Probability: {prob}%")

        st.divider()
        st.subheader("⚠ Risk Report")

        if annual_income < loan_amt:
            st.error("Income is lower than requested loan amount.")
        if credit_score < 600:
            st.error("Credit score is below acceptable level.")
        if debt_to_income_ratio > 0.5:
            st.error("Debt-to-Income ratio is high.")
        if defaults_on_file > 0:
            st.error("Previous loan defaults detected.")

# ================= STATISTICS =================
st.divider()
st.subheader("📊 Application Statistics")

df_stats = pd.read_sql_query("SELECT * FROM predictions", conn)

approved_count = len(df_stats[df_stats["result"] == "Approved"])
rejected_count = len(df_stats[df_stats["result"] == "Rejected"])
total = len(df_stats)
approval_rate = round((approved_count / total) * 100, 2) if total > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("🟢 Approved", approved_count)
col2.metric("🔴 Rejected", rejected_count)
col3.metric("Approval Rate", f"{approval_rate}%")

if total > 0:
    df_stats["date"] = pd.to_datetime(df_stats["timestamp"]).dt.date
    trend = df_stats.groupby(["date", "result"]).size().unstack(fill_value=0)
    st.line_chart(trend)

st.divider()
st.subheader("📂 Live Prediction Dataset")

if total > 0:
    st.dataframe(df_stats.sort_values(by="id", ascending=False), use_container_width=True)

st.markdown(
    "<p style='color:gray;'>⚡ Made By: Rajvi Prasad | 📊 Data Vidwan</p>",
    unsafe_allow_html=True
)
