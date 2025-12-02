"""
Loan Default Study Page
"""

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


# -----------------------------
# Main page function
# -----------------------------
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
    vars_to_study = (get_important_features(df, target_var, PPS_Threshold)
                     .index.tolist())
    df_eda = df.filter([target_var] + vars_to_study)

    st.title("Loan Default Study")

    # Page introduction
    st.info(
        "To help financial institutions better understand what drives "
        "**default risk**, this **Loan Default Study** focuses on "
        "answering **Business Requirement 1**, identifying key borrower "
        "and loan attributes most correlated with default.\n\n"
        "We provide **visual** and **statistical insights** to help business "
        "analysts understand primary drivers of credit risk.\n\n"
        "Distributions and relationships between key features and the target "
        "variable are visualized to uncover patterns distinguishing defaulted "
        "from non-defaulted borrowers."
    )

    st.write("---")

    # Correlation Analysis section
    st.write("## Correlation Analysis")
    st.write(
        "A correlation study was conducted using the **Predictive Power Score "
        "(PPS)**, which detects both linear and non-linear relationships. "
        "The PPS ranges from **0** (no predictive power) to **1** (perfect "
        "predictive power)."
    )

    # Optionally show PPS heatmap
    if st.checkbox("Inspect PPS Correlation Heatmap"):
        heatmap_pps(df)

    # Show PPS table for key features
    st.info(
        "The strongest relationships with the target variable can be detected "
        "for these key features:"
        )
    st.dataframe(get_important_features(df, target_var, PPS_Threshold))

    st.success(
        "In general, correlations with the target are low, indicating that "
        "default behavior is influenced by multiple factors."
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
        "The graphics show that "
        "borrowers who default tend to show the following trends:\n"
        "* Pay rent\n* Have higher interest rates\n"
        "* Higher loan amount relative to income\n\n"
        "Not all defaulting borrowers have all patterns simultaneously; these "
        "factors influence default probability.\n\n"
        "NOTE: As you could see in the PPS Heatmap, "
        "interest rates are strongly correlated with the loan grade. "
        "From a business perspective, having a lower/worse loan grade "
        "implies, that a borrower will have worse loan terms including higher "
        "interest rates. Therefore, having a higher interest rate directly"
        " implies that the borrower is considered more risky."
    )

    # Parallel plot
    st.write("### Parallel Plot")
    st.write(
        "This plot helps explore how variables interact to influence defaults."
        )

    if st.checkbox("Show Parallel Plot"):
        st.write("* Green indicates the profile of defaulted borrowers.")
        parallel_plot_default(df_eda)
        st.success(
            "The parallel categories plot shows the same general patterns "
            "observed in the previous section, but now combined in a single "
            "visualization. It highlights how the key borrower and loan "
            "attributes interact with each other and with default status. "
            "We can see that higher loan amounts relative to income, higher "
            "interest rates, and paying rent tend to be associated with "
            "increased default probability."
        )


# -----------------------------
# Helper functions
# -----------------------------
def get_pps_matrix(df):
    """
    Compute the Predictive Power Score (PPS) matrix for the dataset.
    """
    pps_matrix_raw = pps.matrix(df)
    return pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='y', index='x', values='ppscore')


def heatmap_pps(df, threshold=PPS_Threshold, figsize=(18, 12), font_annot=14):
    """
    Plot a PPS heatmap highlighting scores above a given threshold.
    """
    pps_matrix = get_pps_matrix(df)
    if len(pps_matrix.columns) > 1:
        mask = np.zeros_like(pps_matrix, dtype=bool)
        mask[abs(pps_matrix) < threshold] = True
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pps_matrix,
            annot=True,
            fmt=".3f",
            xticklabels=True,
            yticklabels=True,
            mask=mask,
            cmap='rocket_r',
            annot_kws={"size": font_annot},
            linewidth=0.05,
            linecolor='grey'
        )
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        st.pyplot(fig)


def get_important_features(df, target_var, threshold=PPS_Threshold):
    """
    Return features with PPS above threshold relative to the target.
    """
    pps_matrix = get_pps_matrix(df)
    pps_results = (
        pps_matrix.T[target_var]
        .drop(target_var)
        .loc[lambda x: x.abs() > threshold]
        .sort_values(ascending=False)
        .round(3)
        .to_frame(name="PPS Score")
    )
    pps_results.index.name = None
    return pps_results


def plot_default_level_per_variable(df_eda, target_var):
    """
    Plot distributions of all key variables grouped by default status.

    Categorical variables are plotted with grouped histograms, numerical
    variables with overlaid histograms and marginal boxplots.
    """
    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


def parallel_plot_default(df_eda):
    """
    Display a parallel categories plot for key variables versus loan default.

    Numerical variables are discretized into quartiles for visualization.
    """
    df_eda['loan_int_rate'] = pd.qcut(
        df_eda['loan_int_rate'], q=4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    df_eda['loan_percent_income'] = pd.qcut(
        df_eda['loan_percent_income'], q=4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    fig = px.parallel_categories(
        df_eda,
        color="loan_status",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        width=750,
        height=500
    )
    st.plotly_chart(fig)
