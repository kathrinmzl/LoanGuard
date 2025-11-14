"""
Project Hypotheses & Validation Page.

This module renders the page that evaluates the four key hypotheses
related to factors influencing loan default risk.
"""

import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu
import pandas as pd
from src.plots import plot_categorical, plot_numerical
from src.data_management import load_default_data


def page_project_hypothesis_body():
    """
    Render the Project Hypotheses & Validation page

    Displays and validates the key hypotheses formulated to understand
    factors influencing loan default risk.

    This page helps link domain intuition with data-driven evidence.
    """

    # Use cleaned dataset to ensure statistical tests and distributions
    # reflect real patterns in the data
    df = load_default_data(clean=True, drop_duplicates=True)
    target_var = "loan_status"

    # Page title and introduction
    st.title("Project Hypotheses & Validation")
    st.info(
        "To understand the factors influencing loan default risk, we "
        "formulated four hypotheses grounded in domain knowledge. "
        "Each hypothesis focuses on a feature expected "
        "to affect default probability.\n\n"
        "We validate them using data distributions and statistical "
        "significance tests (Mann–Whitney U or Chi-Square), confirming "
        "whether differences are meaningful."
    )

    # Hypothesis 1
    st.write("## Hypothesis 1")
    st.warning("Higher `loan_amnt` is associated with higher default risk.")
    st.info(
        "Borrowers taking larger loans may face greater repayment burdens, "
        "increasing the likelihood of default."
    )
    st.write("### Validation Result")
    st.success(
        "✅ Confirmed: Defaulted borrowers generally have higher "
        "loan amounts.\n"
        "* The distribution plot supports this observation.\n"
        "* The difference in distributions is statistically significant."
    )

    if st.checkbox("Show Validation: Loan Amount vs. Default"):
        st.write("#### Visualization")
        plot_numerical(df, "loan_amnt", target_var)
        run_mannwhitneyu(df, "loan_amnt", target_var, "higher")

    # Hypothesis 2
    st.write("## Hypothesis 2")
    st.warning("Lower `person_income` is associated with higher default risk.")
    st.info(
        "Borrowers with lower income may have limited financial capacity "
        "to meet repayment obligations."
    )
    st.write("### Validation Result")
    st.success(
        "✅ Confirmed: Defaulted borrowers tend to have lower income levels.\n"
        "* The plot shows a clear difference.\n"
        "* The difference in distributions is statistically significant."
    )

    if st.checkbox("Show Validation: Income vs. Default"):
        st.write("#### Visualization")
        plot_numerical(df, "person_income", target_var)
        run_mannwhitneyu(df, "person_income", target_var, "lower")

    # Hypothesis 3
    st.write("## Hypothesis 3")
    st.warning(
        "Lower `loan_grade` (credit quality) is associated with higher "
        "default risk."
    )
    st.info(
        "A lower loan grade reflects weaker creditworthiness and a higher "
        "assessed lending risk."
    )
    st.write("### Validation Result")
    st.success(
        "✅ Confirmed: Loans with lower grades default more frequently.\n"
        "* The plot shows this trend clearly.\n"
        "* The Chi-Square test confirms the association is statistically "
        "significant."
    )

    if st.checkbox("Show Validation: Loan Grade vs. Default"):
        st.write("#### Visualization")
        plot_categorical(df, "loan_grade", target_var)
        run_chisquare(df, "loan_grade", target_var)

    # Hypothesis 4
    st.write("## Hypothesis 4")
    st.warning(
        "Shorter `person_emp_length` is associated with higher default risk."
    )
    st.info(
        "Borrowers with shorter employment histories may have less income "
        "stability, increasing repayment risk."
    )
    st.write("### Validation Result")
    st.success(
        "✅ Confirmed: Defaulted borrowers typically have shorter employment "
        "histories.\n* The distribution plot supports this.\n"
        "* The Mann–Whitney U test confirms statistical significance."
    )

    if st.checkbox("Show Validation: Employment Length vs. Default"):
        st.write("#### Visualization")
        plot_numerical(df, "person_emp_length", target_var)
        run_mannwhitneyu(df, "person_emp_length", target_var, "lower")

    # Conclusions
    st.write("## Conclusions")
    st.success(
        "All tested hypotheses were confirmed, showing the following:\n\n"
        "- **Domain intuition is supported:** Patterns expected from "
        "financial reasoning are present.\n"
        "- **Features are predictive:** `person_emp_length`, `loan_amnt`, "
        "`person_income`, and `loan_grade` show links to default risk, "
        "making them valuable inputs for predictive models."
    )


# --- Helper Functions --- #
def run_mannwhitneyu(df, col, target_var, direction):
    """Run and display the Mann–Whitney U test for numerical variables."""
    defaulted = df[df[target_var] == 1][col]
    non_defaulted = df[df[target_var] == 0][col]

    stat, p_val = mannwhitneyu(defaulted,
                               non_defaulted,
                               alternative='two-sided')

    st.write("#### Mann–Whitney U Test Result")
    st.write(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        st.info(
            f"✅ Statistically significant difference (p < 0.05): "
            f"defaulted borrowers tend to have **{direction}** `{col}`."
        )
    else:
        st.info(
            f"❌ No statistically significant difference in `{col}` between "
            f"defaulted and non-defaulted borrowers (p ≥ 0.05)."
        )


def run_chisquare(df, col, target_var):
    """Run and display the Chi-Square test for categorical variables."""
    contingency_table = pd.crosstab(df[col], df[target_var])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    st.write("#### Chi-Square Test Result")
    st.write(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        st.info(
            f"✅ Statistically significant association (p < 0.05): "
            f"`{col}` appears to influence default risk."
        )
    else:
        st.info(
            f"❌ No statistically significant association found between "
            f"`{col}` and default risk (p ≥ 0.05)."
        )
