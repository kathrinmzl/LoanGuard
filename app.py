import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_loan_default_study import page_loan_default_study_body
from app_pages.page_predict_input import page_predict_input_body
# from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_model import page_predict_model_body
#from app_pages.page_predict_tenure import page_predict_tenure_body
# from app_pages.page_cluster import page_cluster_body

app = MultiPage(app_name="Loan Guard") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Loan Default Study", page_loan_default_study_body)
# app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("Default Prediction Tool", page_predict_input_body)
app.add_page("ML Model Insights", page_predict_model_body)
# app.add_page("ML: Prospect Tenure", page_predict_tenure_body)
# app.add_page("ML: Cluster Analysis", page_cluster_body)

app.run() # Run the  app
