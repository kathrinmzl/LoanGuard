import streamlit as st


def predict_default(X_live, default_features, default_pipeline_dc_fe, default_pipeline_model):

    # from live data, subset features related to this pipeline
    # could be removed if we dont have cluster analysis
    X_live_default = X_live.filter(default_features)

    # apply data cleaning / feat engine pipeline to live data
    X_live_default_dc_fe = default_pipeline_dc_fe.transform(X_live_default)

    # predict
    default_prediction = default_pipeline_model.predict(X_live_default_dc_fe)
    default_prediction_proba = default_pipeline_model.predict_proba(
        X_live_default_dc_fe)
    
    default_prob = round(default_prediction_proba[0, 1]*100, 2)

    if default_prob < 20:
        risk = 'low'
    elif default_prob < 50:
        risk = 'moderate'
        advice = "Consider lowering the approved loan amount or the interest rate."
    else: 
        risk = 'high'
        advice = "Consider rejecting this loan application."

    st.write(f'### The risk that the loan applicant will default under these conditions is {risk}. ')
    st.info(f'Probability of Default: {default_prob}% ')
    if default_prob >= 20:
        st.success(advice)

    return default_prediction


# def predict_cluster(X_live, cluster_features, cluster_pipeline, cluster_profile):

#     # from live data, subset features related to this pipeline
#     X_live_cluster = X_live.filter(cluster_features)

#     # predict
#     cluster_prediction = cluster_pipeline.predict(X_live_cluster)

#     statement = (
#         f"### The prospect is expected to belong to **cluster {cluster_prediction[0]}**")
#     st.write("---")
#     st.write(statement)

#   	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
#     statement = (
#         f"* Historically, **users in Clusters 0  don't tend to Churn** "
#         f"whereas in **Cluster 1 a third of users churned** "
#         f"and in **Cluster 2 a quarter of users churned**."
#     )
#     st.info(statement)

#   	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
#     statement = (
#         f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
#         f"* Cluster 0 has user without internet, who is a low spender with phone\n"
#         f"* Cluster 1 has user with Internet, who is a high spender with phone\n"
#         f"* Cluster 2 has user with Internet , who is a mid spender without phone"
#     )
#     st.success(statement)

#     # hack to not display index in st.table() or st.write()
#     cluster_profile.index = [" "] * len(cluster_profile)
#     # display cluster profile in a table - it is better than in st.write()
#     st.table(cluster_profile)
