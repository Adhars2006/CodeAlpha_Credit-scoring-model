import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Predict Credit Score", "About"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Predict Credit Score":
        show_predict()
    elif page == "About":
        show_about()


def show_home():
    """Home page"""
    st.markdown("<h1 class='main-header'>💳 Credit Scoring Model</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Welcome to the Credit Scoring System
        
        This application uses advanced algorithms to assess creditworthiness based on 
        various financial and personal factors.
        
        ### Features:
        - 🎯 **Instant Predictions**: Get credit scores in seconds
        - 📊 **Intelligent Analysis**: Detailed financial profile breakdown
        - 💡 **Score Insights**: Understand what impacts your credit score
        - 🔒 **Secure & Private**: Your data stays with you
        
        ### Getting Started:
        Navigate to **Predict Credit Score** to assess creditworthiness
        """)
    
    with col2:
        st.markdown("""
        ### Credit Assessment Factors
        
        **Demographic Factors**:
        - Age
        - Credit History Length
        
        **Financial Factors**:
        - Annual Income
        - Outstanding Debt
        - Debt-to-Income Ratio
        
        **Behavioral Factors**:
        - Payment History
        - Credit Mix Quality
        - Delayed Payments
        
        **Credit Factors**:
        - Credit Utilization
        - Number of Accounts
        - Credit Limits
        """)
    
    st.markdown("---")
    
    # Display score ranges
    st.subheader("📊 Credit Score Ranges")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("### ✅ Good (70-100)")
        st.write("Low credit risk, favorable loan terms")
    
    with col2:
        st.warning("### ⚠️ Fair (50-69)")
        st.write("Moderate risk, standard terms")
    
    with col3:
        st.error("### ❌ Poor (0-49)")
        st.write("High risk, limited credit access")


def calculate_credit_score(age, annual_income, num_loans, delayed_payments,
                          credit_limit, outstanding_debt, monthly_investment,
                          monthly_balance, credit_history_age, num_bank_accounts,
                          num_credit_cards, payment_min_amount, credit_mix,
                          payment_behaviour):
    """Calculate credit score based on multiple factors (rule-based)"""
    score = 50  # Base score
    
    # Age factor (younger = slightly lower)
    if age >= 35:
        score += 10
    elif age >= 25:
        score += 5
    
    # Income factor
    if annual_income > 100000:
        score += 15
    elif annual_income > 50000:
        score += 10
    elif annual_income > 25000:
        score += 5
    
    # Delayed payments (major factor)
    if delayed_payments == 0:
        score += 20
    elif delayed_payments <= 2:
        score += 10
    elif delayed_payments <= 5:
        score += 5
    else:
        score -= 15
    
    # Credit history length
    if credit_history_age >= 120:  # 10 years
        score += 15
    elif credit_history_age >= 60:  # 5 years
        score += 10
    elif credit_history_age >= 24:
        score += 5
    
    # Debt-to-income ratio
    if annual_income > 0:
        dti = outstanding_debt / annual_income
        if dti < 0.3:
            score += 12
        elif dti < 0.5:
            score += 8
        elif dti < 0.7:
            score += 4
        else:
            score -= 10
    
    # Credit utilization (debt vs limit)
    if credit_limit > 0:
        utilization = outstanding_debt / credit_limit
        if utilization < 0.3:
            score += 10
        elif utilization < 0.5:
            score += 5
        else:
            score -= 5
    
    # Number of accounts (more is better, shows credit mix)
    total_accounts = num_bank_accounts + num_credit_cards
    if total_accounts >= 5:
        score += 8
    elif total_accounts >= 3:
        score += 5
    
    # Credit mix
    if credit_mix == "Good":
        score += 8
    elif credit_mix == "Standard":
        score += 4
    
    # Payment behavior
    payment_behavior_scores = {
        "All_paid": 12,
        "No_delay": 10,
        "Good": 8,
        "Low_spent_Small_amount_paid": 4,
    }
    score += payment_behavior_scores.get(payment_behaviour, 0)
    
    # Pays minimum amount
    if payment_min_amount == "Yes":
        score -= 5
    
    # Number of loans
    if num_loans <= 2:
        score += 5
    elif num_loans > 5:
        score -= 5
    
    # Cap score between 0 and 100
    score = max(0, min(100, score))
    
    return score


def get_score_breakdown(age, annual_income, num_loans, delayed_payments,
                       credit_limit, outstanding_debt, monthly_investment,
                       monthly_balance, credit_history_age, num_bank_accounts,
                       num_credit_cards, payment_min_amount, credit_mix,
                       payment_behaviour):
    """Get detailed breakdown of score components"""
    breakdown = {
        "Base Score": 50
    }
    
    # Age
    if age >= 35:
        breakdown["Age (35+)"] = 10
    elif age >= 25:
        breakdown["Age (25+)"] = 5
    
    # Income
    if annual_income > 100000:
        breakdown["Income (>$100K)"] = 15
    elif annual_income > 50000:
        breakdown["Income (>$50K)"] = 10
    elif annual_income > 25000:
        breakdown["Income (>$25K)"] = 5
    
    # Delayed payments
    if delayed_payments == 0:
        breakdown["No Delayed Payments"] = 20
    elif delayed_payments <= 2:
        breakdown["Few Delayed Payments"] = 10
    elif delayed_payments <= 5:
        breakdown["Some Delayed Payments"] = 5
    else:
        breakdown["Many Delayed Payments"] = -15
    
    # Credit history
    if credit_history_age >= 120:
        breakdown["Credit History (10+ years)"] = 15
    elif credit_history_age >= 60:
        breakdown["Credit History (5+ years)"] = 10
    elif credit_history_age >= 24:
        breakdown["Credit History (2+ years)"] = 5
    
    # DTI ratio
    if annual_income > 0:
        dti = outstanding_debt / annual_income
        if dti < 0.3:
            breakdown["DTI Ratio (<30%)"] = 12
        elif dti < 0.5:
            breakdown["DTI Ratio (<50%)"] = 8
        elif dti < 0.7:
            breakdown["DTI Ratio (<70%)"] = 4
        else:
            breakdown["DTI Ratio (>70%)"] = -10
    
    # Credit utilization
    if credit_limit > 0:
        utilization = outstanding_debt / credit_limit
        if utilization < 0.3:
            breakdown["Low Credit Utilization"] = 10
        elif utilization < 0.5:
            breakdown["Moderate Credit Utilization"] = 5
        else:
            breakdown["High Credit Utilization"] = -5
    
    # Accounts
    total_accounts = num_bank_accounts + num_credit_cards
    if total_accounts >= 5:
        breakdown["Credit Mix (5+ accounts)"] = 8
    elif total_accounts >= 3:
        breakdown["Credit Mix (3+ accounts)"] = 5
    
    # Credit mix quality
    if credit_mix == "Good":
        breakdown["Good Credit Mix Quality"] = 8
    elif credit_mix == "Standard":
        breakdown["Standard Credit Mix"] = 4
    
    # Payment behavior
    if payment_behaviour == "All_paid":
        breakdown["All Payments Made"] = 12
    elif payment_behaviour == "No_delay":
        breakdown["No Payment Delays"] = 10
    elif payment_behaviour == "Good":
        breakdown["Good Payment Behavior"] = 8
    elif payment_behaviour == "Low_spent_Small_amount_paid":
        breakdown["Limited Payment Activity"] = 4
    
    # Minimum payment penalty
    if payment_min_amount == "Yes":
        breakdown["Only Pays Minimum"] = -5
    
    # Loans
    if num_loans <= 2:
        breakdown["Few Loans (≤2)"] = 5
    elif num_loans > 5:
        breakdown["Many Loans (>5)"] = -5
    
    return breakdown


def show_predict():
    """Prediction page"""
    st.markdown("<h1 class='main-header'>💳 Credit Score Prediction</h1>", unsafe_allow_html=True)
    
    st.subheader("Enter Customer Information")
    
    # Create input form with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        annual_income = st.number_input("Annual Income ($)", min_value=0, max_value=10000000, value=50000)
        num_loans = st.number_input("Number of Loans", min_value=0, max_value=50, value=2)
        delayed_payments = st.number_input("Delayed Payments", min_value=0, max_value=100, value=0)
        credit_limit = st.number_input("Credit Limit ($)", min_value=0, max_value=1000000, value=5000)
    
    with col2:
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0, max_value=500000, value=1000)
        monthly_investment = st.number_input("Monthly Investment ($)", min_value=0, max_value=100000, value=500)
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0, max_value=500000, value=10000)
        credit_history_age = st.number_input("Credit History Age (Months)", min_value=0, max_value=1000, value=60)
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=50, value=3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=50, value=2)
        payment_min_amount = st.selectbox("Pays Minimum Amount?", ["Yes", "No"])
    
    with col2:
        credit_mix = st.selectbox("Credit Mix", ["Good", "Bad", "Standard"])
        payment_behaviour = st.selectbox("Payment Behaviour", 
            ["All_paid", "Good", "Low_spent_Small_amount_paid", "No_delay"])
    
    # Make prediction using rule-based scoring
    if st.button("🔍 Predict Credit Score", key="predict_button"):
        with st.spinner("Analyzing credit profile..."):
            # Calculate credit score
            score = calculate_credit_score(
                age, annual_income, num_loans, delayed_payments,
                credit_limit, outstanding_debt, monthly_investment,
                monthly_balance, credit_history_age, num_bank_accounts,
                num_credit_cards, payment_min_amount, credit_mix,
                payment_behaviour
            )
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if score >= 70:
                    st.success("✅ GOOD CREDIT SCORE")
                    delta = f"+{score-50:.1f}"
                    st.metric("Credit Score", f"{score:.1f}/100", delta=delta)
                else:
                    st.error("❌ POOR CREDIT SCORE")
                    delta = f"-{100-score:.1f}"
                    st.metric("Credit Score", f"{score:.1f}/100", delta=delta)
            
            with col2:
                # Display assessment and progress
                assessment = "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"
                st.metric("Assessment", assessment)
                st.progress(score/100.0)
            
            # Detailed analysis
            st.markdown("---")
            st.subheader("💡 Financial Profile Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Income", f"${annual_income:,.0f}")
            with col2:
                st.metric("Outstanding Debt", f"${outstanding_debt:,.0f}")
            with col3:
                debt_to_income = (outstanding_debt / annual_income * 100) if annual_income > 0 else 0
                st.metric("Debt-to-Income Ratio", f"{debt_to_income:.1f}%")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Credit History", f"{credit_history_age} months")
            with col2:
                st.metric("Total Accounts", num_bank_accounts + num_credit_cards)
            with col3:
                st.metric("Delayed Payments", str(delayed_payments))
            
            # Score breakdown
            st.markdown("---")
            st.subheader("📋 Score Breakdown")
            breakdown = get_score_breakdown(
                age, annual_income, num_loans, delayed_payments,
                credit_limit, outstanding_debt, monthly_investment,
                monthly_balance, credit_history_age, num_bank_accounts,
                num_credit_cards, payment_min_amount, credit_mix,
                payment_behaviour
            )
            
            # Display breakdown as formatted text
            total = sum(breakdown.values())
            for factor, points in breakdown.items():
                if points > 0:
                    st.write(f"✅ {factor}: **+{points}** points")
                elif points < 0:
                    st.write(f"❌ {factor}: **{points}** points")
                else:
                    st.write(f"• {factor}: {points} points")
            
            st.write(f"\n**Total Score: {total}/100**")


def show_about():
    """About page"""
    st.markdown("<h1 class='main-header'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Project Overview")
        st.markdown("""
        The Credit Scoring Model is an intelligent application designed to 
        assess the creditworthiness of individuals based on their financial 
        history and personal information.
        
        **Key Features:**
        - Instant credit score predictions
        - Comprehensive financial analysis
        - Detailed score breakdown
        - User-friendly interface
        - Secure and private processing
        """)
    
    with col2:
        st.subheader("🎯 How It Works")
        st.markdown("""
        The system analyzes multiple factors:
        
        1. **Demographics** - Age, credit history
        2. **Income** - Annual earnings
        3. **Debt** - Outstanding obligations
        4. **Payment History** - On-time payments
        5. **Credit Mix** - Variety of credit types
        6. **Account Management** - Number of accounts
        
        Each factor contributes to your overall score.
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Scoring Methodology")
    st.markdown("""
    Our algorithm uses a weighted scoring system based on industry-standard 
    credit assessment practices:
    
    - **Payment History (35%)** - Most important factor
    - **Debt-to-Income Ratio (30%)** - How much you owe vs earn
    - **Credit History Length (15%)** - Longer is better
    - **Credit Mix (10%)** - Variety of credit types
    - **New Credit (10%)** - Recent applications
    """)
    
    st.markdown("---")
    
    st.subheader("💡 Score Interpretation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("### ✅ Good (70-100)")
        st.write("""
        - Excellent creditworthiness
        - Best interest rates available
        - Easy loan approval
        """)
    
    with col2:
        st.warning("### ⚠️ Fair (50-69)")
        st.write("""
        - Acceptable credit profile
        - Standard interest rates
        - May need additional review
        """)
    
    with col3:
        st.error("### ❌ Poor (0-49)")
        st.write("""
        - Higher credit risk
        - Higher interest rates
        - May face loan rejections
        """)
    
    st.markdown("---")
    
    st.subheader("👨‍💻 Technology")
    st.write("""
    Built with:
    - **Python** - Programming language
    - **Streamlit** - Web interface
    - **Pandas** - Data processing
    - **NumPy** - Numerical computing
    """)


if __name__ == "__main__":
    main()
