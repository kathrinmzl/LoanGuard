import streamlit as st
from scipy.stats import chi2_contingency
import pandas as pd

from src.plots import plot_categorical, plot_numerical
from src.data_management import load_default_data
from scipy.stats import mannwhitneyu
from feature_engine.outliers import Winsorizer
from feature_engine.imputation import MeanMedianImputer


def page_project_hypothesis_body():

    df = load_default_data()
    
    target_var = "loan_status"

    st.title("Project Hypotheses & Validation")
    
    st.info("To better understand the factors influencing loan default risk, "
             "we formulated four key hypotheses based on domain knowledge and "
             "the available data. Each hypothesis focuses on a variable expected "
             "to impact default probability.\n\n"
             "We validate the hypotheses by examining the distributions and "
             "conduction statistical tests to confirm if the difference in distributions is "
             "statistically significnat or not. "
             
    )
    st.write("---")
    
    # Hypotheses Section
    st.write("## Hypotheses 1")
    st.warning(
        'Higher ``loan_amnt`` is associated with higher default risk.'
        )
    st.info(
        "Rationale: \n\n"
        "Borrowers taking larger loans may face greater repayment burdens, increasing the likelihood of default"
        )
            
    st.write("### Validation Result")
    st.success('We confirm our hypothesis\n\n  '
               ' * The distribution plot shows that the average loan amount '
               'is higher for defaulted borrowers\n'
               ' * Additionally, the difference in distributions is statistically significant')

    if st.checkbox("Validation Loan Amount vs. Default"):
        st.write("#### Visualization")
        plot_numerical(df, "loan_amnt", target_var)
        run_mannwhitneyu(df, "loan_amnt", target_var, "higher")
    
    st.write("---")
    
    st.write("## Hypotheses 2")
    st.warning(
        'Lower ``person_income`` is associated with higher default risk.'
        )
    st.info(
        "Rationale: \n\n"
        "Borrowers with lower income may have limited financial capacity to meet repayment obligations"
        )
            
    st.write("### Validation Result")
    st.success('We confirm our hypothesis\n\n  '
               ' * The distribution plot shows that the average income '
               'is lower for defaulted borrowers\n'
               ' * Additionally, the difference in distributions is statistically significant')

    if st.checkbox("Validation Income vs. Default"):
        st.write("#### Visualization")
        plot_numerical(df, "person_income", target_var)
        run_mannwhitneyu(df, "person_income", target_var, "lower")
        
    st.write("---")
    
    st.write("## Hypotheses 3")
    st.warning(
        'Lower ``loan_grade`` (credit quality) is associated with higher default risk.'
        )
    st.info(
        "Rationale: \n\n"
        "A lower loan grade reflects weaker creditworthiness and higher assessed lending risk"
        )
            
    st.write("### Validation Result")
    st.success('We confirm our hypothesis\n\n  '
               ' * The distribution plot shows that loans with a lower grade '
               'default more often\n'
               ' * Additionally, the association between the two variables is statistically significant')

    if st.checkbox("Validation Loan Grade vs. Default"):
        st.write("#### Visualization")
        plot_categorical(df, "loan_grade", target_var)
        run_chisquare(df, "loan_grade", target_var)
        
    st.write("---")
    
    st.write("## Hypotheses 4")
    st.warning(
        'Shorter `person_emp_length` (employment length) is associated with higher default risk.'
        )
    st.info(
        "Rationale: \n\n"
        "Borrowers with shorter employment histories may experience less income stability, increasing repayment risk"
        )
            
    st.write("### Validation Result")
    st.success('We confirm our hypothesis\n\n  '
               ' * The distribution plot shows that the average employment length '
               'is lower for defaulted borrowers\n'
               ' * Additionally, the difference in distributions is statistically significant')

    if st.checkbox("Validation Employment Length vs. Default"):
        st.write("#### Visualization")
        plot_numerical(df, "person_emp_length", target_var)
        run_mannwhitneyu(df, "person_emp_length", target_var, "lower")


def run_mannwhitneyu(df, col, target_var, direction):
    defaulted = df[df[target_var] == 1]['loan_amnt']
    non_defaulted = df[df[target_var] == 0]['loan_amnt']

    # Run Mann–Whitney U test
    stat, p_val = mannwhitneyu(defaulted, non_defaulted, alternative='greater')

    st.write("#### Result of Mann-Whitney U Test")
    st.write(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        st.info(
            f"The difference in distributions is statistically significant at the 5% significance level. "
            f"This means that defaulted borrowers tend to have {direction} ``{col}``."
        )
    else:
        st.info(
            f"There is no statistically significant difference in ``{col}`` between defaulted and non-defaulted borrowers at the 5% significance level."
        )

def run_chisquare(df, col, target_var):
    contingency_table = pd.crosstab(df[col], df[target_var])

    # Perform Chi-Square test
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    st.write("#### Result of Chi-Square Test")
    st.write(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        st.info(
            f"There is a statistically significant association between the two variables at the 5% significance level. "
            f"This means that ``{col}`` appears to influence default risk."
        )
    else:
        st.info(
            f"There is no statistically significant association found between ``{col}`` and default risk at the 5% significance level."
        )
