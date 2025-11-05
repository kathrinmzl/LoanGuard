import streamlit as st

def page_summary_body():

    st.title("Loan Guard - Project Summary")  # <-- Your app name here
    
    st.write("### Introduction")
    st.info(
        f"In the banking sector, effective credit risk assessment is critical to maintaining financial stability and minimizing losses. "
        f"Loan defaults can lead to significant financial setbacks and reduced liquidity for lending institutions. "
        f"The Loan Guard project aims to help financial institutions better understand the drivers of default risk and proactively identify "
        f"borrowers who are likely to default on their loans.\n\n"
        
        f"From a business perspective, this project helps financial institutions identify high-risk borrowers early, "
        f"make data-driven loan decisions, and provide tailored interventions for at-risk customers."
        f"These insights support profitability and enhance operational efficiency.\n\n"

        f"**The project was created for educational purposes only.**"
    )
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **borrower** is a person who takes out a loan from a financial institution.\n"
        f"* A **loan** is an amount of money borrowed that is\n"
        f"  expected to be paid back with interest.\n"
        f"* A **default** occurs when a borrower fails to make\n"
        f"  scheduled loan payments or meet agreed terms.\n"
        f"* A **defaulted borrower** is a customer who has failed to repay their loan as agreed.\n"
        f"* A **non-default** refers to a borrower who repays\n"
        f"  their loan successfully or continues to pay on time.\n"
    )

    st.write("### Dataset")
    st.info(
        f"The used dataset is publicly available on "
        f"[Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) "
        f"and contains information about individual borrowers and their loans.\n\n"
        f"Each row represents a loan record with personal and financial attributes, such as Age, Income, Employment Length, Loan Amount or Interest Rate. "
        f"The target variable, **`loan_status`**, indicates the loan repayment status. (`0` = non-default, `1` = default)\n\n"
        f"In total, there are **32,581 records and 12 variables**. "
    )
    
    st.write("### Business Requirements")
    
    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 3 business requirements:\n"
        f"1. **Data Insights (Conventional Analysis)**: Identify key borrower and loan attributes that are most correlated with loan default. "
        f"Provide visual and statistical insights to help business analysts understand the primary drivers of credit risk.\n"
        f"2. **Predictive Model (Machine Learning)**: Develop a machine learning model capable of predicting whether a loan applicant is likely to default. "
        f"The system should output a probability of default to support the credit team in decision-making.\n"
        f"3. **Clustering Model (Machine Learning) — Optional**: Group borrowers into risk-based clusters to segment customers by credit behavior "
        f"and improve tailored intervention strategies."
    )
    
    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"📖 For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/kathrinmzl/LoanGuard)."
    )
    
    st.write("### Navigation Guide")
    st.info(
        f"Use the sidebar to switch between pages:\n\n"
        f"1. **Project Summary** – Project overview (this page)\n"
        f"2. **Data Insights** – Answers **BR 1: Data Insights**. Inspect data distributions and feature relationships.\n"
        f"3. **Hypothesis Testing & Validation** – Test key hypotheses about factors influencing default.\n"
        f"4. **Default Prediction** – Answers **BR 2: Predictive Model**. Input borrower info to get default predictions and probabilities. \n"
        f"5. **Classification Model Insights** – Evaluate model metrics and feature importance.\n"
        f"6. **Borrower Clustering Insights (Optional)** – Segment borrowers into risk-based clusters."
    )
