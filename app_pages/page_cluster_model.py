import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_default_data, load_pkl_file

def page_cluster_model_body():
    """
    Streamlit page showing cluster analysis results and profiles for borrowers.
    """

    # Load cluster analysis files and pipeline
    version = 'v2'
    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl"
    )
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_silhouette.png"
    )
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/features_define_cluster.png"
    )
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv"
    )
    cluster_features = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv"
    ).columns.to_list()

    # DataFrame for cluster distribution
    df_cluster_vs_default = load_default_data().filter(['loan_status'], axis=1)
    df_cluster_vs_default['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")

    st.info(
        "* The cluster pipeline using the selected features delivers equivalent performance to the full-feature pipeline.\n"
        "* Average silhouette score: 0.52, indicating reasonable, but not perfectly separated clusters"
    )

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### Features used for clustering")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    cluster_distribution_per_variable(df=df_cluster_vs_default, target='loan_status')

    st.write("#### Most important features defining clusters")
    st.image(features_to_cluster)

    # Updated business interpretation based on new clusters
    st.write("#### Cluster Profile")
    st.info(
        "* Cluster 0: Borrowers with a history of previous defaults, mostly renters, moderate incomes, highest default rate (high-risk segment).\n"
        "* Cluster 1: Borrowers who mostly rent, no default history, lower to mid-range incomes, moderate default rates (medium-risk segment).\n"
        "* Cluster 2: Borrowers who primarily have mortgages, no default history, higher incomes, rarely default (low-risk segment)."
        )
    st.warning(
        "* The cluster profile allows the credit team to identify risk characteristics beyond the predicted probability of default.\n"
        "* Potential actions: higher scrutiny or additional guarantees for Cluster 0, cautious lending for Cluster 1, standard approval for Cluster 2."
        )

    # Show cluster profile table
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)


def cluster_distribution_per_variable(df, target):
    """
    Displays bar and line plots showing cluster distribution across a target variable.
    """

    df_bar_plot = df.groupby(["Clusters", target]).size().reset_index(name="Count")
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    st.write(f"#### Cluster distribution across {target} levels")
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

