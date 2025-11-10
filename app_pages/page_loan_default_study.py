import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import ppscore as pps

from src.data_management import load_default_data
from src.plots import plot_categorical, plot_numerical

# Seaborn style
sns.set_style("whitegrid")

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
target_var = "loan_status"
PPS_Threshold = 0.04


def page_loan_default_study_body():
    """
    Render the Loan Default Study page to answer Business Requirement 1.

    Loads the dataset, identifies important features using PPS, and displays
    correlation heatmaps, feature distributions, and parallel plots to analyze
    the drivers of loan default.
    """

    # Load dataset
    df = load_default_data()

    # Select important features based on PPS threshold
    vars_to_study = get_important_features(df, target_var,
                                          PPS_Threshold).index.tolist()
    df_eda = df.filter([target_var] + vars_to_study)

    st.title("Loan Default Study")

    # Page introduction
    st.info(
        f"To help financial institutions better understand what drives "
        f"**default risk**, this **Loan Default Study** focuses on answering "
        f"**Business Requirement 1**, identifying key borrower and loan "
        f"attributes most correlated with default.\n\n"
        f"We provide **visual** and **statistical insights** to help business "
        f"analysts understand primary drivers of credit risk.\n\n"
        f"Distributions and relationships between key features and the target "
        f"variable are visualized to uncover patterns distinguishing defaulted "
        f"from non-defaulted borrowers."
    )

    # Optional data inspection
    if st.checkbox("Inspect Loan Default Dataset"):
        st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        st.write(df.head(10))

    st.write("---")

    # Correlation Analysis section
    st.write("## Correlation Analysis")
    st.write(
        f"A correlation study was conducted using **Predictive Power Score "
        f"(PPS)**, which detects both linear and non-linear relationships. "
        f"The PPS ranges from **0** (no predictive power) to **1** (perfect "
        f"predictive power)."
    )

    # Optionally show PPS heatmap
    if st.checkbox("Inspect PPS Correlation Heatmap"):
        heatmap_pps(df)

    # Show PPS table for key features
    st.info("Strongest relationships with the target variable:")
    st.dataframe(get_important_features(df, target_var, PPS_Threshold))

    st.success(
        "Correlations with the target are generally low, indicating default "
        "behavior is influenced by multiple factors."
    )
    st.write("---")

    # Feature distributions section
    st.write("## Relationships between Key Features and Loan Default")
    st.write(
        "Explore distributions of key borrower and loan features and their "
        "relation to loan default outcome. This complements the correlation "
        "analysis and identifies trends or potential risk factors."
    )

    if st.checkbox("Key feature distributions divided by default levels"):
        plot_default_level_per_variable(df_eda, target_var)

    # Summary of general trends
    st.success(
        f"Borrowers who default tend to show the following trends:\n"
        f"* Pay rent\n* Have higher interest rates\n"
        f"* Higher loan amount relative to income\n\n"
        f"Not all defaulting borrowers have all patterns simultaneously; these "
        f"factors influence default probability."
    )

    # Parallel plot
    st.write("### Parallel Plot")
    st.write("This plot helps explore how variables interact to influence defaults.")

    if st.checkbox("Show Parallel Plot"):
        st.write("* Green indicates the profile of defaulted borrowers.")
        parallel_plot_default(df_eda)
        st.success(
            "The plot highlights interactions of key borrower and loan attributes "
            "with default status."
        )


# -----------------------------
# Helper functions
# -----------------------------

def get_pps_matrix(df):
    """Compute the PPS matrix for the dataset."""
    pps_matrix_raw = pps.matrix(df)
    return pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='y', index='x', values='ppscore')


def heatmap_pps(df, threshold=PPS_Threshold, figsize=(18, 12), font_annot=14):
    """Plot PPS heatmap highlighting scores above the threshold."""
    pps_matrix = get_pps_matrix(df)
    if len(pps_matrix.columns) > 1:
        mask = np.zeros_like(pps_matrix, dtype=bool)
        mask[abs(pps_matrix) < threshold] = True
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pps_matrix, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='rocket_r', annot_kws={"size": font_annot},
                    linewidth=0.05, linecolor='grey')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        st.pyplot(fig)


def get_important_features(df, target_var, threshold=PPS_Threshold):
    """Return features with PPS above threshold with respect to target."""
    pps_matrix = get_pps_matrix(df)
    pps_results = (
        pps_matrix.T[target_var]
        .drop(target_var)
        .loc[lambda x: x.abs() > threshold]
        .sort_values(ascending=False)
        .to_frame(name="PPS Score")
    )
    pps_results.index.name = None
    return pps_results


def plot_default_level_per_variable(df_eda, target_var):
    """Plot all key variables grouped by default/non-default borrowers."""
    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


def parallel_plot_default(df_eda):
    """Display parallel plot for key variables versus loan default."""
    # Discretize numerical features into quartiles
    df_eda['loan_int_rate'] = pd.qcut(df_eda['loan_int_rate'], q=4,
                                      labels=['Low', 'Medium', 'High', 'Very High'])
    df_eda['loan_percent_income'] = pd.qcut(df_eda['loan_percent_income'], q=4,
                                           labels=['Low', 'Medium', 'High', 'Very High'])

    fig = px.parallel_categories(
        df_eda, color="loan_status",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        width=750, height=500
    )
    st.plotly_chart(fig)
