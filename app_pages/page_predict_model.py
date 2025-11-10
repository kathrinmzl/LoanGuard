import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_predict_model_body():
    """
    Render the ML Model Insights page

    Displays information about the trained Loan Default Prediction model,
    including.

    This page helps technical and business users understand how well the
    model identifies potential loan defaulters and whether it meets the
    recall and F1-score goals required for deployment.
    """

    version = 'v2'

    # Load saved model assets
    default_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_default/{version}/clf_pipeline_data_cleaning_feat_eng.pkl')
    default_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_default/{version}/clf_pipeline_model.pkl")
    default_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_default/{version}/features_importance.png")

    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_default/{version}/y_test.csv").values

    # Page title and success criteria
    st.title("ML Model Insights")

    st.info(
        f"### Model Objective\n"
        f"The model aims to identify potential **loan defaulters** early.\n\n"
        f"Success is defined by meeting these criteria (both on train & test sets):\n"
        f"- **Recall (Default) ≥ 0.75** — minimize false negatives "
        f"(don’t miss high-risk borrowers)\n"
        f"- **F1 Score (Default) ≥ 0.60** — maintain balance between recall and precision"
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
        f"**Performance Summary:**\n\n"
        f"- **Train Set (Default class):** Recall = 0.82, F1 = 0.80 ✅\n"
        f"- **Test Set (Default class):** Recall = 0.80, F1 = 0.61 ✅\n\n"
        f"Both metrics meet the defined success criteria. "
        f"The model performs consistently across training and test data.\n\n"
        f"**Observations:**\n"
        f"* Recall remains high, ensuring most high-risk borrowers are detected.\n"
        f"* F1 score shows a moderate drop from train to test (0.80 → 0.61), "
        f"reflecting expected variability due to resampling and unseen data.\n"
        f"* The drop is acceptable and indicates **no severe overfitting**.\n\n"
        f"**Business Insight:**\n"
        f"* The model effectively prioritizes recall, aligning with the goal "
        f"of minimizing missed defaulters.\n"
        f"* Slight precision trade-off is acceptable in exchange for better risk control.\n"
        f"* Overall, the pipeline demonstrates good generalization and is "
        f"suitable for deployment in a real credit risk monitoring context."
    )
