import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Hybrid Credit Risk Prediction System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Hybrid Credit Risk Prediction System")
st.markdown("""
This system combines **loan repayment prediction** and **mobile money behavioral risk**.

**Final Risk = 0.7 × Default Probability + 0.3 × Behavior Risk**
""")

# -----------------------------
# Load models and preprocessors
# -----------------------------
model_a = joblib.load("model_a.pkl")
preprocessor_a = joblib.load("preprocessor.pkl")

model_b = joblib.load("model_b.pkl")
scaler_b = joblib.load("scaler_b.pkl")
risk_scaler_b = joblib.load("risk_scaler_b.pkl")


# -----------------------------
# Risk band function
# -----------------------------
def risk_band(score):
    if score < 0.20:
        return "Low Risk"
    elif score < 0.40:
        return "Moderate Risk"
    elif score < 0.60:
        return "High Risk"
    else:
        return "Very High Risk"


# -----------------------------
# Loan applicant inputs
# -----------------------------
st.header("1. Loan Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Retired", "Student"])

with col2:
    annual_income = st.number_input("Annual Income", min_value=0.0, value=40000.0)
    debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.01, 1.00, 0.18)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0)
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Debt consolidation", "Other", "Car", "Home", "Education", "Business", "Medical", "Vacation"]
    )

with col3:
    interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 12.5)
    loan_term = st.selectbox("Loan Term (Months)", [36, 60])
    installment = st.number_input("Installment", min_value=0.0, value=450.0)
    grade_subgrade = st.selectbox(
        "Grade/Subgrade",
        [
            "A1", "A2", "A3", "A4", "A5",
            "B1", "B2", "B3", "B4", "B5",
            "C1", "C2", "C3", "C4", "C5",
            "D1", "D2", "D3", "D4", "D5",
            "E1", "E2", "E3", "E4", "E5",
            "F1", "F2", "F3", "F4", "F5"
        ]
    )

st.header("2. Credit History Information")

col4, col5, col6 = st.columns(3)

with col4:
    num_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=5)
    total_credit_limit = st.number_input("Total Credit Limit", min_value=0.0, value=50000.0)

with col5:
    current_balance = st.number_input("Current Balance", min_value=0.0, value=20000.0)
    delinquency_history = st.number_input("Delinquency History", min_value=0, value=2)

with col6:
    public_records = st.number_input("Public Records", min_value=0, value=0)
    num_of_delinquencies = st.number_input("Number of Delinquencies", min_value=0, value=2)


# -----------------------------
# Mobile money behavior inputs
# -----------------------------
st.header("3. Mobile Money Behavioral Information")

col7, col8, col9 = st.columns(3)

with col7:
    transaction_count = st.number_input("Transaction Count", min_value=1, value=10)
    total_amount = st.number_input("Total Transaction Amount", min_value=0.0, value=500000.0)
    avg_amount = st.number_input("Average Transaction Amount", min_value=0.0, value=50000.0)
    max_amount = st.number_input("Maximum Transaction Amount", min_value=0.0, value=150000.0)

with col8:
    min_amount = st.number_input("Minimum Transaction Amount", min_value=0.0, value=5000.0)
    std_amount = st.number_input("Transaction Amount Variation", min_value=0.0, value=30000.0)
    avg_oldbalance = st.number_input("Average Old Balance", min_value=0.0, value=300000.0)
    avg_newbalance = st.number_input("Average New Balance", min_value=0.0, value=250000.0)

with col9:
    unique_destinations = st.number_input("Unique Recipients", min_value=1, value=5)
    avg_step = st.number_input("Average Transaction Step", min_value=0.0, value=100.0)
    std_step = st.number_input("Transaction Step Variation", min_value=0.0, value=20.0)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Hybrid Credit Risk"):

    # -----------------------------
    # Model A input
    # -----------------------------
    loan_input = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education_level": education_level,
        "annual_income": annual_income,
        "employment_status": employment_status,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "interest_rate": interest_rate,
        "loan_term": loan_term,
        "installment": installment,
        "grade_subgrade": grade_subgrade,
        "num_of_open_accounts": num_of_open_accounts,
        "total_credit_limit": total_credit_limit,
        "current_balance": current_balance,
        "delinquency_history": delinquency_history,
        "public_records": public_records,
        "num_of_delinquencies": num_of_delinquencies
    }])

    loan_processed = preprocessor_a.transform(loan_input)

    paid_back_prob = model_a.predict_proba(loan_processed)[:, 1][0]
    default_prob = 1 - paid_back_prob

    # -----------------------------
    # Model B input
    # -----------------------------
    momo_input = pd.DataFrame([{
        "transaction_count": transaction_count,
        "total_amount": total_amount,
        "avg_amount": avg_amount,
        "max_amount": max_amount,
        "min_amount": min_amount,
        "std_amount": std_amount,
        "avg_oldbalance": avg_oldbalance,
        "avg_newbalance": avg_newbalance,
        "unique_destinations": unique_destinations,
        "avg_step": avg_step,
        "std_step": std_step
    }])

    # Engineered features used during Model B training
    momo_input["amount_per_transaction"] = momo_input["total_amount"] / momo_input["transaction_count"]
    momo_input["amount_range"] = momo_input["max_amount"] - momo_input["min_amount"]
    momo_input["destination_ratio"] = momo_input["unique_destinations"] / momo_input["transaction_count"]

    # Very important: arrange columns exactly as the scaler saw during training
    momo_input = momo_input[scaler_b.feature_names_in_]

    momo_scaled = scaler_b.transform(momo_input)

    anomaly_score = model_b.decision_function(momo_scaled)
    raw_behavior_risk = -anomaly_score

    behavior_risk = risk_scaler_b.transform(
        raw_behavior_risk.reshape(-1, 1)
    ).flatten()[0]

    behavior_risk = max(0, min(1, behavior_risk))

    # -----------------------------
    # Final hybrid risk
    # -----------------------------
    final_risk = (0.7 * default_prob) + (0.3 * behavior_risk)
    band = risk_band(final_risk)

    # -----------------------------
    # Results
    # -----------------------------
    st.markdown("---")
    st.header("Prediction Results")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric("Paid Back Probability", f"{paid_back_prob:.2%}")

    with r2:
        st.metric("Default Probability", f"{default_prob:.2%}")

    with r3:
        st.metric("Behavior Risk", f"{behavior_risk:.2%}")

    with r4:
        st.metric("Final Risk Score", f"{final_risk:.2%}")

    st.subheader(f"Risk Band: {band}")

    if band == "Low Risk":
        st.success("Recommendation: Applicant appears safe for loan approval.")
    elif band == "Moderate Risk":
        st.info("Recommendation: Applicant may be approved with caution.")
    elif band == "High Risk":
        st.warning("Recommendation: Applicant should undergo further review.")
    else:
        st.error("Recommendation: Applicant is very risky and may be rejected.")

    with st.expander("View Loan Input Data"):
        st.dataframe(loan_input)

    with st.expander("View Mobile Money Input Data"):
        st.dataframe(momo_input)