import streamlit as st


# def predict_default(X_live, default_features, default_pipeline_dc_fe, default_pipeline_model):

#     # from live data, subset features related to this pipeline
#     # could be removed if we dont have cluster analysis
#     X_live_default = X_live.filter(default_features)

#     # apply data cleaning / feat engine pipeline to live data
#     X_live_default_dc_fe = default_pipeline_dc_fe.transform(X_live_default)

#     # predict
#     default_prediction = default_pipeline_model.predict(X_live_default_dc_fe)
#     default_prediction_proba = default_pipeline_model.predict_proba(
#         X_live_default_dc_fe)
    
#     default_prob = round(default_prediction_proba[0, 1]*100, 2)

#     if default_prob < 20:
#         risk = 'low'
#     elif default_prob < 50:
#         risk = 'moderate'
#         advice = "Consider lowering the approved loan amount."
#     else: 
#         risk = 'high'
#         advice = "Consider rejecting this loan application."

#     st.write(f'### The risk that the loan applicant will default under these conditions is {risk}. ')
#     st.info(f'Probability of Default: {default_prob}% ')
#     if default_prob >= 20:
#         st.success(advice)

#     return default_prediction


# def predict_cluster(X_live, cluster_features, cluster_pipeline, cluster_profile):

#     # from live data, subset features related to this pipeline
#     X_live_cluster = X_live.filter(cluster_features)

#     # predict
#     cluster_prediction = cluster_pipeline.predict(X_live_cluster)
#     # st.write(cluster_prediction)

#     statement = (
#         f"### The prospect is expected to belong to **cluster {cluster_prediction[0]}**")
#     st.write("---")
#     st.write(statement)

#   	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
#     statement = (
#         f"Historically, **borrowers in Cluster 0 showed low default rates** "
#         f"whereas in **Cluster 1 a third of borrowers defaulted**."
#     )
#     st.info(statement)

#   	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
#     statement = (
#         f"The cluster profile interpretation allows us to label the clusters in the following fashion:\n"
#         f"* Cluster 0 includes borrowers with mortgages, higher income, and lower interest rates\n"
#         f"* Cluster 1 primarily includes renters with lower income and higher interest rates\n"
#         f"Note: Home ownership is the most important feature defining the clusters"
#     )
#     st.success(statement)

#     # hack to not display index in st.table() or st.write()
#     cluster_profile.index = [" "] * len(cluster_profile)
#     # display cluster profile in a table - it is better than in st.write()
#     st.table(cluster_profile)
    
def predict_default_and_cluster(X_live, 
                                default_features, default_pipeline_dc_fe, default_pipeline_model, 
                                cluster_features, cluster_pipeline, cluster_profile):
    """
    Predict default risk and cluster for a live prospect, and give a combined recommendation.
    """

    # Default Prediction
    X_live_default = X_live[default_features]
    X_live_default_dc_fe = default_pipeline_dc_fe.transform(X_live_default)
    default_pred = default_pipeline_model.predict(X_live_default_dc_fe)
    default_pred_proba = default_pipeline_model.predict_proba(X_live_default_dc_fe)
    
    default_prob = round(default_pred_proba[0, 1] * 100, 2)
    
    st.write("---")
    st.write(f"### There is a {default_prob}% probability that this borrower will default")

    # Cluster Prediction
    X_live_cluster = X_live[cluster_features]
    cluster_pred = cluster_pipeline.predict(X_live_cluster)[0]

    st.write("---")
    st.write(f"### The prospect is expected to belong to **Cluster {cluster_pred}**")
    
    # Provide cluster context
    if cluster_pred == 0:
        st.info("Historically, borrowers in Cluster 0 showed low default rates (~12%).")
        st.warning("Profile: Borrowers who mostly own a mortgage, have higher income, and receive lower interest rates")
    elif cluster_pred == 1:
        st.info("Historically, borrowers in Cluster 1 showed higher default rates (~30%).")
        st.warning("Profile: Borrowers who mostly rent, have lower income, and face higher interest rates")
    st.write(f"Note: Home ownership is the most important feature defining the clusters")
    # Show cluster profile table
    cluster_profile.index = [" "] * len(cluster_profile)  # hide index for display
    st.table(cluster_profile)
    
    # Combined Business Recommendation
    recommendation = ""
    if default_prob >= 50 and cluster_pred == 1:
        recommendation = "High risk: Consider rejecting or restructuring the loan."
    elif default_prob >= 50 and cluster_pred == 0:
        recommendation = "Moderate risk: Approve only with additional guarantees or lower loan amount."
    elif default_prob < 50 and cluster_pred == 1:
        recommendation = "Moderate risk: Approve with caution; consider lowering interest rate exposure or loan amount."
    else:  # default_prob <50 and cluster 0
        recommendation = "Low risk: Approve the loan."

    st.write("---")
    st.write(f"### Recommended action")
    st.success(recommendation)

    # Return predictions for further use if needed
    return default_pred, cluster_pred, default_prob, recommendation

