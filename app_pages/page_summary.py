import streamlit as st

def page_summary_body():
    """
    Render the Project Summary page for the Loan Guard app.
    
    This page provides an overview of the project, dataset, business 
    requirements, key terminology, and navigation instructions.
    """
    
    # Page title
    st.title("Loan Guard - Project Summary") 
    
    # Introduction section
    st.write("### Introduction")
    st.info(
        f"In the banking sector, effective credit risk assessment is critical "
        f"to maintaining financial stability and minimizing losses. "
        f"Loan defaults can lead to significant financial setbacks and "
        f"reduced liquidity for lending institutions. "
        f"The Loan Guard project aims to help financial institutions better "
        f"understand the drivers of default risk and proactively identify "
        f"borrowers who are likely to default on their loans.\n\n"
        
        f"From a business perspective, this project helps financial "
        f"institutions identify high-risk borrowers early, make data-driven "
        f"loan decisions, and provide tailored interventions for at-risk "
        f"borrowers. These insights support profitability and enhance "
        f"operational efficiency.\n\n"

        f"**The project was created for educational purposes only.**"
    )
    
    # Project terms and jargon
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **borrower** is a person who takes out a loan from a financial "
        f"  institution.\n"
        f"* A **loan** is an amount of money borrowed that is expected to be "
        f"  paid back with interest.\n"
        f"* A **default** occurs when a borrower fails to make scheduled loan "
        f"  payments or meet agreed terms.\n"
        f"* A **defaulted borrower** is a borrower who has failed to repay "
        f"  their loan as agreed.\n"
        f"* A **non-default** refers to a borrower who repays their loan "
        f"  successfully or continues to pay on time.\n"
    )
    
    # Dataset section
    st.write("### Dataset")
    st.info(
        f"The used dataset is publicly available on "
        f"[Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) "
        f"and contains information about individual borrowers and their loans.\n\n"
        f"Each row represents a loan record with personal and financial "
        f"attributes, such as Age, Income, Employment Length, Loan Amount, "
        f"or Interest Rate. The target variable, **`loan_status`**, indicates "
        f"the loan repayment status (`0` = non-default, `1` = default).\n\n"
        f"In total, there are **32,581 records and 12 variables**."
    )
    
    # Business requirements section
    st.write("### Business Requirements")
    
    # Copied from README file
    st.success(
        f"The project has 3 business requirements:\n"
        f"1. **Data Insights (Conventional Analysis)**: Identify key borrower "
        f"and loan attributes most correlated with loan default. Provide visual "
        f"and statistical insights to help business analysts understand the "
        f"primary drivers of credit risk.\n"
        f"2. **Predictive Model (Machine Learning)**: Develop a machine "
        f"learning model capable of predicting whether a loan applicant is "
        f"likely to default. The system should output a probability of default "
        f"to support the credit team in decision-making.\n"
        f"3. **Clustering Model (Machine Learning) — Optional**: Group "
        f"borrowers into risk-based clusters to segment borrowers by credit "
        f"behavior and improve tailored intervention strategies."
    )
    
    # Link to README file
    st.write(
        f"📖 For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/kathrinmzl/LoanGuard)."
    )
    
    # Navigation guide section
    st.write("### Navigation Guide")
    st.info(
        f"Use the sidebar to switch between pages:\n\n"
        f"1. **Project Summary** – Project overview (this page)\n"
        f"2. **Loan Default Study** – Answers **BR 1: Data Insights**. Inspect "
        f"data distributions and feature relationships.\n"
        f"3. **Project Hypotheses & Validation** – Test key hypotheses about "
        f"factors influencing default.\n"
        f"4. **Default Prediction Tool** – Answers **BR 2: Predictive Model**. "
        f"Input borrower info to get default predictions and probabilities.\n"
        f"5. **ML Model Insights** – Evaluate model metrics and "
        f"feature importance.\n"
        # f"6. **Borrower Clustering Insights (Optional)** – Segment borrowers "
        # f"into risk-based clusters."
    )
