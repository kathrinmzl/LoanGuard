import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_default_data, load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_predict_model_body():

    version = 'v1'
    # load needed files
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

    st.title("ML Model Insights")
    # display pipeline training summary conclusions
    # In the conclusions summary, we are primarily interested in documenting the ML performance,
    # so technical users that visit this page can quickly grasp the outcome from training that pipeline.
    st.info(
        f"TO DO UPDATE * The pipeline was tuned aiming at least 0.80 Recall on 'Yes default' class, "
        f"since we are interested in this project in detecting a potential defaulter. \n"
        f"* The pipeline performance on train and test set is 0.90 and 0.85, respectively."
    )

    # show pipelines
    st.write("---")
    st.write("## There are 2 ML Pipelines arranged in series:")

    st.write(" * The first is responsible for data cleaning and feature engineering")
    st.write(default_pipe_dc_fe)

    st.write("* The second is for feature scaling and modelling")
    st.write(default_pipe_model)

    # show feature importance plot
    st.write("---")
    st.write("## Feature Importance")
    st.write("The features the model was trained on and their importance are as follows:")
    st.write(X_train.columns.to_list())
    st.image(default_feat_importance)

    # We don't need to apply dc_fe pipeline, since X_train and X_test
    # were already transformed in the jupyter notebook (Predict Customer default.ipynb)

    # evaluate performance on train and test set
    st.write("---")
    st.write("## Model Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=default_pipe_model,
                    label_map=["No Default", "Default"])
    
    st.write("---")
    st.write("## Interpretation")
    st.success(
        f"**Confusion Matrix (Test Set):**\n"
        f"- ✅ 4084 borrowers correctly predicted as **No Default**\n"
        f"- ✅ 1064 borrowers correctly predicted as **Default**\n"
        f"- ❌ 302 borrowers actually defaulted but predicted as **No Default** (false negatives — most costly!)\n"
        f"- ❌ 1034 borrowers did not default but predicted as **Default** (false positives)\n\n"
        f"**Classification Report (Test Set):**\n"
        f"- **Precision (Default): 0.51** → Only about half of flagged borrowers actually default\n"
        f"- **Recall (Default): 0.78** → Most defaulters are correctly identified, critical for risk management\n"
        f"- **F1-Score (Default): 0.61** → Moderate balance between precision and recall\n"
        f"- **Accuracy: 0.79** → Overall, the model correctly predicts 79% of borrowers\n\n"
        f"**Comparison to Train Set Results:**\n"
        f"* The model generalizes well: accuracy and recall drop only slightly (~1%) from train to test, indicating no severe overfitting.\n"
        f"* Precision for Default drops on the test set (0.81 → 0.51), showing the model is more conservative in flagging high-risk borrowers on unseen data.\n"
        f"* Recall remains high, which is desirable to catch potential defaulters.\n"
        f"* Slight underfitting in Default precision is offset by maintaining high recall, aligning with the business goal of minimizing missed high-risk borrowers.\n\n"
        f"**Business Interpretation:**\n"
        f"* The small metric shift between train and test indicates the model is robust and reliable for decision-making.\n"
        f"* High recall ensures most risky borrowers are detected; the drop in precision is an acceptable trade-off.\n"
        f"* Overall, the model balances loss prevention with borrower impact, supporting the credit team's risk management strategy."
    )
