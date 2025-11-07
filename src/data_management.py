import streamlit as st
import pandas as pd
import numpy as np
import joblib
from feature_engine.outliers import Winsorizer
from feature_engine.imputation import MeanMedianImputer

@st.cache_data
def load_default_data():
    df = pd.read_csv("outputs/datasets/collection/LoanDefaultData.csv")
    imputer = MeanMedianImputer(imputation_method='median', variables=['person_emp_length', 'loan_int_rate'])
    df = imputer.fit_transform(df)
    winsorizer = Winsorizer(capping_method='iqr', fold=5, tail='right', variables=["person_income","person_emp_length"])
    df = winsorizer.fit_transform(df)
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)