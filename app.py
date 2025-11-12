"""
Main entry point for the "Loan Guard" Streamlit application.

Initializes the MultiPage app, registers all pages, and launches the app.
"""

from app_pages.multipage import MultiPage

# Import the scripts for each page of the app
from app_pages.page_summary import page_summary_body
from app_pages.page_loan_default_study import page_loan_default_study_body
from app_pages.page_predict_input import page_predict_input_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_model import page_predict_model_body
from app_pages.page_cluster_model import page_cluster_model_body

# Initialize the Streamlit MultiPage app
app = MultiPage(app_name="Loan Guard")

# Add pages to the app
app.add_page("Project Summary", page_summary_body)
app.add_page("Loan Default Study", page_loan_default_study_body)
app.add_page("Project Hypotheses & Validation", page_project_hypothesis_body)
app.add_page("Default Prediction Tool", page_predict_input_body)
app.add_page("ML Prediction Model Insights", page_predict_model_body)
app.add_page("ML Cluster Model Insights", page_cluster_model_body)

# Execute the app
app.run()
