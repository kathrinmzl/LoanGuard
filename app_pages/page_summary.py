"""
Project Summary Page for the Loan Guard Streamlit application.

This module renders the summary page, providing an overview of the project,
dataset, business requirements, terminology, and navigation guidance for
users exploring the Loan Guard credit risk analysis app.
"""

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
        "In the banking sector, effective credit risk assessment is "
        "critical to maintaining financial stability and minimizing "
        "losses. Loan defaults can lead to significant financial setbacks "
        "and reduced liquidity for lending institutions. The Loan Guard "
        "project aims to help financial institutions better understand the "
        "drivers of default risk and proactively identify borrowers who "
        "are likely to default on their loans.\n\n"

        "From a business perspective, this project helps financial "
        "institutions identify high-risk borrowers early, make data-driven "
        "loan decisions, and provide tailored interventions for at-risk "
        "borrowers. These insights support profitability and enhance "
        "operational efficiency.\n\n"

        "**The project was created for educational purposes only.**"
    )

    # Project terms and jargon
    st.info(
        "**Project Terms & Jargon**\n"
        "* A **borrower** is a person who takes out a loan from a financial "
        "  institution.\n"
        "* A **loan** is an amount of money borrowed that is expected to be "
        "  paid back with interest.\n"
        "* A **default** occurs when a borrower fails to make scheduled loan "
        "  payments or meet agreed terms.\n"
        "* A **defaulted borrower** is a borrower who has failed to repay "
        "  their loan as agreed.\n"
    )

    # Dataset section
    st.write("### Dataset")
    st.info(
        "The used dataset is publicly available on "
        "[Kaggle](https://www.kaggle.com/datasets/laotse/"
        "credit-risk-dataset/data) and contains information about "
        "individual borrowers and their loans.\n\n"
        "Each row represents a loan record with personal and financial "
        "attributes, such as Age, Income, Employment Length, Loan Amount, "
        "or Interest Rate. The target variable, **`loan_status`**, "
        "indicates the loan repayment status (`0` = non-default, "
        "`1` = default).\n\n"
        "In total, there are **32,581 records and 12 variables**."
    )

    # Business requirements section
    st.write("### Business Requirements")

    st.success(
        "The project has 3 business requirements:\n"
        "1. **Data Insights (Conventional Analysis)**: Identify key "
        "borrower and loan attributes most correlated with loan default. "
        "Provide visual and statistical insights to help business analysts "
        "understand the primary drivers of credit risk.\n"
        "2. **Predictive Model (Machine Learning)**: Develop a machine "
        "learning model capable of predicting whether a loan applicant is "
        "likely to default. The system should output a probability of "
        "default to support the credit team in decision-making.\n"
        "3. **Clustering Model (Machine Learning)**: Group borrowers into "
        "risk-based clusters to segment borrowers by credit behavior and "
        "improve tailored intervention strategies."
    )

    # Link to README file
    st.write(
        "📖 For additional information, please visit and **read** the "
        "[Project README file](https://github.com/kathrinmzl/LoanGuard)."
    )

    # Navigation guide section
    st.write("### Navigation Guide")
    st.info(
        "Use the sidebar to switch between pages:\n\n"
        "1. **Project Summary** – Project overview (this page)\n"
        "2. **Loan Default Study** – Answers **BR 1: Data Insights**. "
        "Inspect data distributions and feature relationships.\n"
        "3. **Project Hypotheses & Validation** – Test key hypotheses about "
        "factors influencing default.\n"
        "4. **Default Prediction Tool** – Answers **BR 2: Predictive Model** "
        "and **BR 3: Clustering Model**. Input borrower info to get default "
        "predictions and assigned clusters.\n"
        "5. **ML Classification Model Insights** – Evaluate model metrics and "
        "feature importance.\n"
        "6. **ML Borrower Clustering Insights** – Evaluate model metrics and "
        "cluster profiles."
    )
