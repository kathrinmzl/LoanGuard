import streamlit as st
import pandas as pd
from src.data_management import load_default_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_default)#,
   # predict_tenure,
    #predict_cluster)

# answers business requirement 2

def page_predict_input_body():

    # load predict default files
    version = 'v2'
    default_pipe_dc_fe = load_pkl_file(
        f'outputs/ml_pipeline/predict_default/{version}/clf_pipeline_data_cleaning_feat_eng.pkl')
    default_pipe_model = load_pkl_file(
        f"outputs/ml_pipeline/predict_default/{version}/clf_pipeline_model.pkl")
    default_features = (pd.read_csv(f"outputs/ml_pipeline/predict_default/{version}/X_train.csv")
                      .columns
                      .to_list()
                      )

    # # load cluster analysis files
    # version = 'v1'
    # cluster_pipe = load_pkl_file(
    #     f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    # cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
    #                     .columns
    #                     .to_list()
    #                     )
    # cluster_profile = pd.read_csv(
    #     f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")

    st.title("Default Prediction Tool")
    
    st.info(
        f"This page implements Business Requirement 2: 'Develop a machine learning model capable  "
        f"of predicting whether a loan applicant is likely to default. The system should output a "
        f"probability of default to support the credit team in decision-making.'\n\n"
        f"The model evaluates a borrower's probability of default given the key inputs of "
        f"`loan_amnt`, `person_income` and `loan_int_rate`. For a given borrower's income, "
        f"the credit team can determine whether the requested loan amount is likely to be "
        f"repayable at the selected interest rate.\n\n"
        f"If the predicted probability of default is moderate or high under the current terms, "
        f"the credit team can take action — for example, lower the approved loan amount, reduce the "
        f"interest rate or decline the loan application — to reduce default risk. These interventions help balance loss prevention "
        f"with borrower impact and improve overall financial stability."
    )

    st.write("---")

    # Generate/save Live Data
    
    # use to check which inout variables you will need to implement
    # check_variables_for_UI(default_features)
    X_live = DrawInputsWidgets()

    '''
    Answer business logic:
    The client is interested in determining whether or not a given prospect will default. 
    If so, the client is interested to know when. 
    In addition, the client is interested in learning from which cluster this prospect 
    will belong in the customer base.
    '''
    # predict on live data
    if st.button("Run Predictive Analysis"):
        predict_default(
            X_live, default_features, default_pipe_dc_fe, default_pipe_model)

        # predict_cluster(X_live, cluster_features,
        #                 cluster_pipe, cluster_profile)


# def check_variables_for_UI( default_features):
#     # combines all features and displays the unique values.
    
#     import itertools

#     # The widgets inputs are the features used in all pipelines (tenure, default, cluster)
#     # We combine them only with unique values
#     combined_features = set(
#         # list(
#         #     itertools.chain(tenure_features, default_features, cluster_features)
#         # )
#         list(
#             itertools.chain(default_features)
#         )
#     )
#     st.write(
#         f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():
    '''
    return a DataFrame with 1 row containing the prospect’s information.
    The interactive widgets will feed the data values to the DataFrame in real time.
      
    Then, we need to populate the widgets. If it is a categorical variable, we list the available options.
    If it is a numerical variable, we will set the initial value as the median value from
    the variable, the minimum widget value as 0.4 of the min from that variable and the maximum widget
    value as x2 the max value from that variable. 
    
    The decision to display the median value and the 0.4
    and 2.0 are arbitrary, in your project you could set other values if you would like. To populate
    the proper initial values, we load the dataset, so we can extract the values from it afterwards.
    '''

    # load dataset
    df = load_default_data()
    percentageMin = 0.1

    # we create input widgets only for 6 features
    col1, col2, col3 = st.columns(3)
    # col5, col6, col7, col8 = st.columns(4)

    # We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result
    # {'PhoneService', 'MonthlyCharges', 'PaymentMethod', 'InternetService', 'Contract', 'OnlineBackup'}

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values
    

    with col1:
        feature = "person_income"
        st_widget = st.number_input(
            label="Person Income",
            min_value=int(df[feature].min()*percentageMin),
            max_value=300000, 
            value=int(df[feature].median()),
            step=10
        )
    X_live[feature] = st_widget
    
    with col2:
        feature = "loan_amnt"
        st_widget = st.number_input(
            label="Loan Amount",
            min_value=int(df[feature].min()*percentageMin),
            max_value=300000,
            value=int(df[feature].median()),
            step=10
        )
        
    X_live[feature] = st_widget

    with col3:
        feature = "loan_int_rate"
        st_widget = st.number_input(
            label="Loan Interest Rate",
            min_value=round(float(df[feature].min() * percentageMin), 2),
            max_value=round(float(df[feature].max() * 1.5), 2),
            value=round(float(df[feature].median()), 2),
            step=0.01,              # allows increments of 0.01
            format="%.2f"           # always show two decimal places
        )
    X_live[feature] = st_widget  # Add widget to live dataframe

    # show live data table 
    # st.write(X_live)

    return X_live
