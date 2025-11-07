import plotly.express as px
import numpy as np
import pandas as pd
# from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_default_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import plotly.express as px

import ppscore as pps

# Ignore FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Answers business requirement 1

target_var = "loan_status"
PPS_Threshold = 0.04

# Body
def page_loan_default_study_body():

    # load data
    df = load_default_data()
    

    # hard copied from churned customer study notebook
    # vars_to_study = ['Contract', 'InternetService',
    #                  'OnlineSecurity', 'TechSupport', 'tenure']
    
    vars_to_study = get_important_features(df, target_var, PPS_Threshold).index.tolist()

    df_eda = df.filter([target_var]+vars_to_study)
    
    st.title("Loan Default Study")

    st.info(
        f"To help financial institutions better understand what drives **default risk**, "
        f"this **Loan Default Study** focuses on answering **Business Requirement 1**, "
        f"which concerns identifying key borrower and loan attributes that are most correlated "
        f"with loan default.\n\n"
        f"We provide **visual** and **statistical insights** to help business analysts understand "
        f"the primary drivers of credit risk. This is primarily achieved through **correlation analysis** "
        f"to identify borrower and loan characteristics most strongly linked to default behavior.\n\n"
        f"Additionally, we **visualize distributions and relationships** between key features and "
        f"the target variable to uncover patterns that distinguish defaulted from non-defaulted borrowers."
    )

    # Optionally inspect data
    if st.checkbox("Inspect Loan Default Dataset"):
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Find the first 10 rows below.")
        st.write(df.head(10))

    st.write("---")

    # Correlation Study 
    st.write("## Correlation Analysis")
    st.write(
        f"A correlation study was conducted to identify relationships between variables and the target feature. "
        f"To demonstrate the results, we use the **Predictive Power Score (PPS)**, which detects both linear and non-linear relationships between two variables. "
        f"The PPS ranges from **0** (no predictive power) to **1** (perfect predictive power). "
        f"\n\nThe heatmap below visualizes the predictive strength between all variables in the dataset. "
        f"To address **Business Requirement 1**, we focus on the relationships between the independent variables and the target variable **`loan_status`**."
    )
    # Heatmap
    if st.checkbox("Inspect PPS Correlation Heatmap"):
        heatmap_pps(df)
    # PPS Table
    st.info(
        f"The strongest relationships with the target variable can be identified for the following features: "
    )
    st.dataframe(get_important_features(df, target_var, PPS_Threshold))
    st.success(
        f"The correlation analysis reveals that correlations with the target are generally low. "
        f"These findings are consistent with business logic and indicate that default behavior is influenced "
        f"by a combination of factors rather than a single feature."
        )
    st.write("---")
    
    st.write("## Relationships between Key Features and Loan Default")
    st.write(
        f"Now we can explore the distributions of the identified key borrower and loan features "
        f"and how they relate to the loan default outcome. By visualizing these patterns, "
        f"we complement the correlation analysis and identify trends, imbalances, or "
        f"potential risk factors for loan defaults."
    )
    # Individual plots per variable
    if st.checkbox("Key feature distributions divided by default levels"):
        plot_default_level_per_variable(df_eda, target_var)
    
    # Text based on "02 - Churned Customer Study" notebook - "Conclusions and Next steps" section
    st.success(
        f"The analysis shows that borrowers who default tend to show the following general trends:\n"
        f"* Pay rent\n"
        f"* Have higher interest rates\n"
        f"* Have a higher loan amount relative to their income\n\n"
        f"This doesn't mean all defaulting customers have all these patterns at the same time; "
        f"these are just factors that influence the probability of default."
     )
    
    """
    Add: 
    > NOTE: Interest rates are strongly correlated with the loan grade. Having a lower/worse loan grade implies that a borrower 
    will have worse loan terms including higher interest rates. Therefore, having a higher interest rate directly implies that the borrower is more risky.
    """

    st.write('### Parallel Plot')
    st.write('The following parallel plot further helps to explore how these variables interact together to influence default outcomes.')
    # Parallel plot
    if st.checkbox("Show Parallel Plot"):
        st.write(
            f"* Information in green indicates the profile from a defaulted borrower")
        parallel_plot_default(df_eda)
        st.success("The plot highlights how key borrower and loan attributes interact with each other and with default status. "
                   "We can see that higher loan amounts relative to income, higher interest rates and paying rent "
                   "tend to be associated with increased default probability.")


# Plots 
def get_pps_matrix(df):
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(columns='y', index='x', values='ppscore')
    return pps_matrix
    
    
def heatmap_pps(df, threshold=PPS_Threshold, figsize=(18, 12), font_annot=14):
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
    pps_matrix = get_pps_matrix(df)
    pps_results = (
        pps_matrix.T[target_var]   # select target column
        .drop(target_var)                   # remove self-correlation
        .loc[lambda x: x.abs() > threshold]  # apply threshold
        .sort_values(ascending=False)       # optional: sort
        .to_frame(name="PPS Score")         # convert Series to DataFrame with new column name
    )
    pps_results.index.name = None
    return pps_results
    
    
# function created using "02 - Churned Customer Study" notebook code - "Variables Distribution by Churn" section
def plot_default_level_per_variable(df_eda, target_var):
    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


def plot_categorical(df, col, target_var):
    # Create interactive countplot
    fig = px.histogram(
        df,
        x=col,
        color=target_var,
        barmode="group",  # shows bars side by side like hue in seaborn
        category_orders={col: df[col].value_counts().index.tolist()},  # preserves order
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        title_text=f"{col}",
        title_x=0.5,
        xaxis_title="",
        yaxis_title="Count",
        legend_title=target_var
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
def plot_numerical(df, col, target_var):
    fig = px.histogram(
        df,
        x=col,
        color=target_var,
        barmode="overlay",       # overlay bars for each target class
        histnorm='',             # raw counts, can also use 'percent'
        marginal="box",
        color_discrete_sequence=px.colors.qualitative.Set2,
        nbins=50                # adjust for resolution
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color="black"
    )

    fig.update_layout(
        title_text=f"{col}",
        title_x=0.5,
        xaxis_title="",
        yaxis_title="Count",
        legend_title=target_var
    )

    st.plotly_chart(fig, use_container_width=True)


# function created using "02 - Churned Customer Study" notebook code - Parallel Plot section
def parallel_plot_default(df_eda):
    
    # Discretize numerical features into quartile bins
    df_eda['loan_int_rate'] = pd.qcut(df_eda['loan_int_rate'], q=4,
        labels=['Low', 'Medium', 'High', 'Very High'])

    df_eda['loan_percent_income'] = pd.qcut(df_eda['loan_percent_income'], q=4,
        labels=['Low', 'Medium', 'High', 'Very High'])

    fig = px.parallel_categories(
        df_eda, color="loan_status", 
        color_continuous_scale=px.colors.diverging.RdYlGn,
        width=750, height=500)
    # we use st.plotly_chart() to render, in notebook is fig.show()
    st.plotly_chart(fig)
