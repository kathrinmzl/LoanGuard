import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_predict_model_body():
    """
    Render the ML Prediction Model Insights page.

    This page presents details about the supervised classification model
    developed to predict loan default risk. It shows the ML pipeline,
    key features, evaluation metrics, and interpretation of results.
    """

    version = 'v2'

    # Load saved model assets
    default_pipe_dc_fe = load_pkl_file(
        f"outputs/ml_pipeline/predict_default/{version}/"
        f"clf_pipeline_data_cleaning_feat_eng.pkl"
    )
    default_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_default/{version}/"
        f"clf_pipeline_model.pkl"
    )
    default_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_default/{version}/"
        f"features_importance.png"
    )

    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/y_train.csv"
    ).values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/y_test.csv"
    ).values

    # Page title and success criteria
    st.title("ML Prediction Model Insights")

    st.info(
        "### Model Objective\n"
        "The model aims to identify potential **loan defaulters** early.\n\n"
        "Success is defined by meeting these criteria "
        "(both on train & test sets):\n"
        "- **Recall (Default) ≥ 0.75** — minimize false negatives "
        "(don’t miss high-risk borrowers)\n"
        "- **F1 Score (Default) ≥ 0.60** — maintain balance between "
        "recall and precision"
    )

    # Pipelines overview
    st.write("## ML Pipelines Overview")

    st.write("* **Pipeline 1:** Data Cleaning and Feature Engineering")
    st.write(default_pipe_dc_fe)

    st.write("* **Pipeline 2:** Feature Scaling and Model Training")
    st.write(default_pipe_model)

    # Feature importance
    st.write("## Feature Importance")
    st.write("The model was trained using the following features:")
    st.write(X_train.columns.to_list())
    st.image(default_feat_importance)

    # Model evaluation
    st.write("## Model Performance")
    clf_performance(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        pipeline=default_pipe_model,
        label_map=["No Default", "Default"]
    )

    # Interpretation and business insights
    st.write("## Interpretation & Business Relevance")
    st.success(
        "**Performance Summary:**\n\n"
        "- **Train Set (Default class):** Recall = 0.80, F1 = 0.80 ✅\n"
        "- **Test Set (Default class):** Recall = 0.78, F1 = 0.62 ✅\n\n"
        "Both metrics meet the defined success criteria.\n\n"
        "**Observations:**\n"
        "* Recall remains high across both datasets, ensuring most high-risk "
        "borrowers are detected.\n"
        "* The F1 score shows a moderate drop from train to test "
        "(0.80 → 0.62), mainly driven by lower precision on unseen data.\n"
        "* This drop is expected and does not indicate severe overfitting, "
        "as recall remains stable and well above the threshold.\n\n"
        "**Business Insight:**\n"
        "* The model effectively prioritizes recall, supporting the goal of "
        "minimizing missed defaulters.\n"
        "* The precision trade-off is acceptable for credit risk management, "
        "where capturing high-risk applicants is more important than "
        "occasional false positives.\n"
        "* Overall, the pipeline meets business performance requirements and "
        "provides reliable support for credit risk decision-making."
    )
