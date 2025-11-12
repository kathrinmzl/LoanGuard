"""
Predictive analytics utilities.

This module integrates the trained machine learning pipeline for loan
default prediction with the customer segmentation (clustering) model.
It provides real-time risk assessment, cluster allocation, and a combined
business recommendation for new borrower data.
"""

import streamlit as st


def predict_default_and_cluster(
        X_live, default_features, default_pipeline_dc_fe,
        default_pipeline_model, cluster_features, cluster_pipeline,
        cluster_profile):
    """
    Predict default risk and cluster for live data, and provide a combined
    business recommendation. Supports 3 clusters.
    """

    st.write("---")
    st.write("### Predicting Default Risk")

    # Default Prediction
    X_live_default = X_live[default_features]
    X_live_default_dc_fe = default_pipeline_dc_fe.transform(X_live_default)
    default_pred = default_pipeline_model.predict(X_live_default_dc_fe)
    default_pred_proba = default_pipeline_model.predict_proba(
        X_live_default_dc_fe)
    default_prob = round(default_pred_proba[0, 1] * 100, 2)

    st.success(
        f"There is a **{default_prob}% probability** that this borrower "
        f"will default."
    )

    st.write("---")
    st.write("### Predicting Cluster Membership")

    # Cluster Prediction
    X_live_cluster = X_live[cluster_features]
    cluster_pred = cluster_pipeline.predict(X_live_cluster)[0]

    st.success(
        f"The prospect is expected to belong to **Cluster {cluster_pred}**"
    )

    # Cluster context for the new 3-cluster setup
    cluster_context = {
        0: {
            "default_rate": "38%",
            "profile": (
                "Previous default history, mostly renters, moderate incomes, "
                "higher-risk segment"
            )
        },
        1: {
            "default_rate": "26%",
            "profile": (
                "No previous default history, mostly renters, lower to "
                "mid-range incomes, moderate default rate, middle-risk "
                "segment"
            )
        },
        2: {
            "default_rate": "9%",
            "profile": (
                "No previous default history, primarily mortgage holders, "
                "higher incomes, rarely default, lowest-risk segment"
            )
        }
    }

    if cluster_pred in cluster_context:
        st.info(
            f"Historically, borrowers in Cluster {cluster_pred} showed a "
            f"default rate of "
            f"{cluster_context[cluster_pred]['default_rate']}."
        )
        st.warning(
            f"Cluster Profile: "
            f"{cluster_context[cluster_pred]['profile']}"
        )

    st.write("Note: Income is the least important feature defining the "
             "clusters.")

    # Display cluster profile table
    cluster_profile.index = [" "] * len(cluster_profile)  # hide index
    st.table(cluster_profile)

    st.write("---")
    st.write("### Combined Business Recommendation")

    # Combined recommendation logic
    if default_prob >= 50 and cluster_pred == 0:
        recommendation = (
            "High risk: Consider rejecting or restructuring the loan."
        )
    elif default_prob >= 50 and cluster_pred in [1, 2]:
        recommendation = (
            "Moderate risk: Approve only with additional guarantees or "
            "lower loan amount."
        )
    elif default_prob < 50 and cluster_pred == 0:
        recommendation = (
            "Moderate risk: Approve with caution; consider lowering loan "
            "amount or interest rate exposure."
        )
    elif default_prob < 50 and cluster_pred == 1:
        recommendation = (
            "Low-to-moderate risk: Standard approval with monitoring "
            "recommended."
        )
    else:  # default_prob < 50 and cluster 2
        recommendation = "Low risk: Approve the loan."

    st.success(recommendation)

    return default_pred, cluster_pred, default_prob, recommendation
