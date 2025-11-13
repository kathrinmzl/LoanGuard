import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_default_data, load_pkl_file


def page_cluster_model_body():
    """
    Render ML Cluster Model Insights page.

    This page summarizes the results of the unsupervised clustering analysis.
    It shows the ML pipeline, feature importance, silhouette performance,
    cluster profiles, and business relevance.
    """

    # Load cluster analysis files and pipeline
    version = 'v2'
    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl"
    )
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/"
        f"clusters_silhouette.png"
    )
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/"
        f"features_define_cluster.png"
    )
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv"
    )
    cluster_features = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv"
    ).columns.to_list()

    # DataFrame for cluster distribution
    df_cluster_vs_default = load_default_data(
        drop_duplicates=True
    ).filter(['loan_status'], axis=1)
    df_cluster_vs_default['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")

    st.info(
        "### Model Objective\n"
        "The clustering model aims to group borrowers into distinct "
        "**risk-based segments** based on their financial and credit "
        "characteristics.\n\n"
        "This helps credit teams understand borrower behavior, tailor "
        "interventions, and design more targeted lending strategies.\n\n"
        "Success is defined by meeting these criteria:\n"
        "- **Average Silhouette Score ≥ 0.45** — ensures well-separated "
        "and meaningful clusters\n"
        "- **Cluster Interpretability** — each cluster should have a "
        "distinct financial and behavioral profile aligned with business "
        "understanding of borrower risk"
    )

    st.write("## ML Pipeline Overview")
    st.write(cluster_pipe)

    st.write("## Feature Importance")
    st.write("The clusters were defined using the following features:")
    st.write(cluster_features)
    st.image(features_to_cluster)

    st.write("## Model Performance")
    st.write("### Clusters Silhouette Plot")
    st.image(cluster_silhouette)
    st.info(
        "* Number of clusters chosen: 3\n"
        "* Average Silhouette Score: 0.52 ≥ 0.45"
    )

    cluster_distribution_per_variable(
        df=df_cluster_vs_default, target='loan_status'
    )

    # Updated business interpretation based on new clusters
    st.write("## Cluster Profile")
    st.info(
        "* Cluster 0: Borrowers with previous defaults, mostly renters, "
        "moderate incomes, and highest default rate (high-risk segment).\n"
        "* Cluster 1: Borrowers who rent, no default history, lower to "
        "mid-range incomes, moderate default rates (medium-risk segment).\n"
        "* Cluster 2: Borrowers with mortgages, higher incomes, no default "
        "history, lowest default rate (low-risk segment)."
    )
    st.warning(
        "* The cluster profile allows the credit team to identify risk "
        "characteristics beyond the predicted probability of default.\n"
        "* Potential actions: higher scrutiny for Cluster 0, cautious "
        "lending for Cluster 1, standard approval for Cluster 2."
    )

    # Show cluster profile table
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)

    # Interpretation and business insights
    st.write("## Interpretation & Business Relevance")
    st.success(
        "**Performance Summary:**\n\n"
        "- **Average Silhouette Score:** 0.52 ✅\n"
        "- The score meets the defined success criterion (≥ 0.45), "
        "indicating a satisfactory level of cluster separation and "
        "cohesion.\n"
        "- The number of clusters is low and clusters are well-defined "
        "and distinct in their borrower profiles.\n\n"
        "**Business Insight:**\n"
        "* The segmentation supports targeted credit strategies — for "
        "example, closer monitoring or stricter approval thresholds for "
        "high-risk clusters.\n"
        "* The clusters are interpretable and actionable, providing "
        "meaningful differentiation among borrower types.\n"
        "* Overall, the clustering model fulfills the project’s analytical "
        "and business requirements, supporting both credit risk evaluation "
        "and strategic decision-making."
    )


def cluster_distribution_per_variable(df, target):
    """
    Displays bar and line plots showing cluster distribution across a target
    variable.
    """

    df_bar_plot = (df
                   .groupby(["Clusters", target])
                   .size()
                   .reset_index(name="Count")
                   )
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    st.write(f"## Cluster distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    st.plotly_chart(fig)

    df_relative = (df.groupby(["Clusters", target])
                   .size()
                   .unstack(fill_value=0)
                   .apply(lambda x: 100 * x / x.sum(), axis=1)
                   .stack()
                   .reset_index(name='Relative Percentage (%)')
                   .sort_values(by=['Clusters', target])
                   )
    df_relative.columns = ['Clusters', target, 'Relative Percentage (%)']

    st.write(f"#### Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig)
