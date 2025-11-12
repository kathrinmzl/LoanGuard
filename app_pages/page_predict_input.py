"""
Default Prediction Tool Page.

This module renders the page where users can input borrower data
and obtain both a predicted probability of loan default and a cluster
assignment. It provides a combined business recommendation based
on both predictions.
"""

import streamlit as st
import pandas as pd
from src.data_management import load_default_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_default_and_cluster
)


def page_predict_input_body():
    """
    Streamlit page for predicting loan default probability and borrower
    cluster. Provides a combined business recommendation based on both
    predictions.
    """

    # Load default prediction pipeline
    version = 'v2'
    default_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_default/{version}/'
        'clf_pipeline_data_cleaning_feat_eng.pkl'
    )
    default_pipe_model = load_pkl_file(
        f'outputs/ml_pipeline/predict_default/{version}/clf_pipeline_model.pkl'
    )
    default_features = pd.read_csv(
        f'outputs/ml_pipeline/predict_default/{version}/X_train.csv'
    ).columns.to_list()

    # Load cluster analysis pipeline
    cluster_pipe = load_pkl_file(
        f'outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl'
    )
    cluster_features = pd.read_csv(
        f'outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv'
    ).columns.to_list()
    cluster_profile = pd.read_csv(
        f'outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv'
    )

    # Page title and explanation
    st.title("Default Prediction Tool")
    st.info(
        "This page implements Business Requirement 2: 'Develop a machine "
        "learning model capable of predicting whether a loan applicant is "
        "likely to default and segmenting borrowers into risk clusters.'\n\n"
        "The system provides two outputs for a live borrower:\n"
        "1. A **probability of default** based on key inputs such as "
        "`loan_amnt`, `person_income`, and `loan_int_rate`.\n"
        "2. A **cluster assignment**, which segments borrowers into groups "
        "with historically similar default behavior and financial "
        "profiles.\n\n"
        "The credit team can assess repayment likelihood and also consider "
        "the borrower's cluster to understand broader risk characteristics "
        "(e.g., prior defaults, home ownership, income level).\n\n"
        "If the predicted probability of default is moderate or high, "
        "and/or the borrower belongs to a higher-risk cluster, the credit "
        "team can take targeted actions — e.g., lower the approved loan "
        "amount, require additional guarantees, or decline the application — "
        "to reduce default risk."
    )
    st.warning(
        "**Note:** The thresholds for business recommendations (low, "
        "moderate, high risk) should be determined according to the "
        "institution's risk appetite."
    )

    st.write("---")

    # Draw input widgets for live data
    # Use helper function to check which input variables you will need
    # to implement
    # check_variables_for_UI(default_features, cluster_features)
    X_live = DrawInputsWidgets()

    # Predict default and cluster for the live prospect
    if st.button("Run Predictive Analysis"):
        predict_default_and_cluster(
            X_live,
            default_features,
            default_pipe_dc_fe,
            default_pipe_model,
            cluster_features,
            cluster_pipe,
            cluster_profile
        )


# Helper: show combined features for UI reference
def check_variables_for_UI(default_features, cluster_features):
    import itertools
    combined_features = set(itertools.chain(default_features,
                                            cluster_features))
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n"
        f"{combined_features}"
    )


# Draw input widgets for live data
def DrawInputsWidgets():
    """
    Generates a DataFrame with a single row containing user inputs from
    interactive widgets. Categorical variables are selectboxes, numerical
    variables are number inputs with sensible defaults.
    """
    df = load_default_data()
    percentage_min = 0.1  # used to scale min values

    # Layout: 3 columns per row
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    # Create empty DataFrame for live input
    X_live = pd.DataFrame([], index=[0])

    # Numerical Inputs
    with col1:
        feature = "person_income"
        X_live[feature] = st.number_input(
            label="Person Income",
            min_value=int(df[feature].min() * percentage_min),
            max_value=300000,
            value=int(df[feature].median()),
            step=10
        )
    with col2:
        feature = "loan_amnt"
        X_live[feature] = st.number_input(
            label="Loan Amount",
            min_value=int(df[feature].min() * percentage_min),
            max_value=300000,
            value=int(df[feature].median()),
            step=10
        )
    with col3:
        feature = "loan_int_rate"
        X_live[feature] = st.number_input(
            label="Loan Interest Rate",
            min_value=round(float(df[feature].min() * percentage_min), 2),
            max_value=round(float(df[feature].max() * 1.5), 2),
            value=round(float(df[feature].median()), 2),
            step=0.01,
            format="%.2f"
        )

    # Categorical Inputs
    with col4:
        feature = "person_home_ownership"
        X_live[feature] = st.selectbox(
            label="Person Home Ownership",
            options=df[feature].unique()
        )
    with col5:
        feature = "cb_person_default_on_file"
        X_live[feature] = st.selectbox(
            label="Default on File",
            options=df[feature].unique()
        )

    # Optional: display the live input table
    # st.write(X_live)

    return X_live
