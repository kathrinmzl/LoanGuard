"""
Model Evaluation Utilities for Loan Default Prediction.

This module provides reusable Streamlit-compatible functions for evaluating
classification model performance.

These functions are used throughout the Loan Guard application to ensure
consistent reporting and visualization of model results across pages.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_and_report(X, y, pipeline, label_map):
    """
    Generate and display confusion matrix and classification report.
    """
    # Make predictions
    y_pred = pipeline.predict(X)

    # Confusion Matrix
    st.write("#### Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"{label}" for label in label_map],
        columns=[f"{label}" for label in label_map]
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='rocket_r',
        annot_kws={"size": 12},
        linewidth=0.5,
        linecolor='grey'
    )
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    st.pyplot(fig, use_container_width=False)

    # Classification Report
    st.write("#### Classification Report")
    report = classification_report(
        y, y_pred, target_names=label_map, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()

    # Move accuracy to a separate summary section
    accuracy = report_df.loc["accuracy", "precision"]
    report_df = report_df.drop("accuracy", errors="ignore")

    st.dataframe(report_df.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:.0f}"
    }))

    # Display accuracy separately
    st.info(f"**Overall Accuracy:** {accuracy:.2%}")


def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    """
    Display classification performance on train and test datasets.
    """
    st.write("### Train Set")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    st.write("### Test Set")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)
