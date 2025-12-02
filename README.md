# Loan Guard

Developed by [kathrinmzl](https://www.github.com/kathrinmzl)

[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/kathrinmzl/LoanGuard)](https://www.github.com/kathrinmzl/LoanGuard/commits/main)
![GitHub last commit](https://img.shields.io/github/last-commit/kathrinmzl/LoanGuard?color=red)
![GitHub language count](https://img.shields.io/github/languages/count/kathrinmzl/LoanGuard?color=yellow)
![GitHub top language](https://img.shields.io/github/languages/top/kathrinmzl/LoanGuard?color=green)
[![badge](https://img.shields.io/badge/deployment-Heroku-purple)](https://loan-guard-c4aee35f5523.herokuapp.com/)

In the banking sector, effective credit risk assessment is critical for maintaining financial stability and minimizing losses. Loan defaults can lead to significant financial setbacks and reduced liquidity for lending institutions. The Loan Guard project aims to help financial institutions proactively identify borrowers who are likely to default and understand the key factors driving default risk.

In addition, the project incorporates borrower segmentation through clustering, which groups borrowers with similar financial profiles and historical behavior. This combined approach allows lenders not only to predict default probabilities but also to tailor risk management strategies and business decisions based on the characteristics of different borrower segments.

The project was created for educational purposes only.

[Live page on Heroku](https://loan-guard-c4aee35f5523.herokuapp.com/)


## Dataset Content

The used dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and contains information about individual borrowers and their loan characteristics. Each row represents a loan record, including both personal and financial attributes that may influence the likelihood of default. The dataset provides a comprehensive overview of borrower profiles, such as age, income, home ownership or employment details, as well as loan-specific features like loan amount, interest rate and purpose.  

In total, the dataset includes **32,581 records and 12 variables**. The target variable, **`loan_status`**, indicates whether a borrower has defaulted on their loan (`1`) or successfully repaid it (`0`). The target distribution is **imbalanced toward non-default cases**, reflecting real-world lending scenarios where most borrowers do not default. This dataset enables predictive modeling to identify patterns and risk factors associated with loan default.

| Variable | Description | Role | Data Type | Units / Possible Values |
|-----------|-------------|------|------------|--------------------------|
| `person_age` | Age of the borrower | Feature | int64 | Years |
| `person_income` | Annual income of the borrower | Feature | float64 | USD  |
| `person_home_ownership` | Type of home ownership | Feature | object | RENT, OWN, MORTGAGE, OTHER |
| `person_emp_length` | Length of employment | Feature | float64 | Years |
| `loan_intent` | Purpose of the loan | Feature | object | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| `loan_grade` | Loan grade assigned by lender | Feature | object | A, B, C, D, E, F, G |
| `loan_amnt` | Loan amount requested | Feature | float64 | USD |
| `loan_int_rate` | Interest rate applied to the loan | Feature | float64 | Percentage |
| `loan_percent_income` | Loan amount as a percentage of annual income | Feature | float64 | Ratio |
| `cb_person_default_on_file` | Whether the person has previously defaulted | Feature | object | Y, N |
| `cb_person_cred_hist_length` | Length of credit history | Feature | int64 | Years |
| `loan_status` | Loan default flag | Target | int64 | 0 = No Default, 1 = Default |

>**NOTE:**
<br><br>
When I initially started working on this project, I used a different dataset from [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default). After attempting to build meaningful prediction and clustering models, I decided to switch to a new dataset.
<br><br>
The previous dataset was highly synthetic, with all variables being uniformly distributed and showing very little correlation, both between features and with the target variable. Uniform distributions are particularly challenging for predictive modeling and clustering because they lack natural variability and concentration of values. Consequently, there are few meaningful patterns, groupings, or relationships for the models to learn from.
<br><br>
As a result, it was very difficult to build a predictive model with good performance metrics and without overfitting. I experimented with several approaches to improve model performance and reduce overfitting, including hyperparameter tuning and binning numerical variables, but none led to satisfactory results. Furthermore, during the cluster analysis, the results did not correspond to any recognizable borrower groups or risk profiles, limiting the usefulness of the analysis.
<br><br>
Therefore, I decided to switch to the current dataset. Although it required more extensive data cleaning and transformation, it produces models with stronger performance and revealed meaningful, interpretable clusters. Overall, business interpretability and analytical insight were significantly improved.


## Project Terms & Jargon 
- A **borrower** is a person who takes out a loan from a financial institution.  
- A **loan** is an amount of money borrowed that is expected to be paid back with interest.  
- A **default** occurs when a borrower fails to make scheduled loan payments or meet the agreed repayment terms.  
- A **defaulted borrower** is a borrower who has failed to repay their loan as agreed and is classified as being in default.  


## Business Requirements
From a business perspective, this project supports the strategic goals of a financial institution such as:

- Improving risk management by identifying high-risk applicants early. 
- Enhancing profitability through optimized loan approval decisions.
- Increasing borrower trust and operational efficiency by offering fair, data-driven credit evaluations.
- Enabling personalized loan offerings and proactive interventions for at-risk borrowers (e.g., adjusted payment plans or counseling).

Ultimately, this project aligns predictive analytics with the bank’s long-term objective of balancing growth with financial stability.

To achieve the outlined objectives, the project will focus on the following key requirements.

**Business Requirement 1: Data Insights (Conventional Analysis)**
- Identify key borrower and loan attributes that are most correlated with loan default.
Provide visual and statistical insights to help business analysts understand the primary drivers of credit risk.

**Business Requirement 2: Classification Model (Machine Learning)**
- Develop a machine learning model capable of predicting whether a loan applicant is likely to default. The system should output a probability of default to support the credit team in decision-making.

**Business Requirement 3: Clustering Model (Machine Learning)**
- Group borrowers into risk-based clusters to segment borrowers by credit behavior and improve tailored intervention strategies.


## Hypotheses and how to validate them?
To better understand the factors influencing loan default risk,  four key hypotheses were formulated based on domain knowledge and the available data. Each hypothesis focuses on a variable expected to impact default probability.

| Hypothesis | Rationale | Validation |
|------------|------------|-------------|
| **H1:** Higher `loan_amnt` is associated with higher default risk | Borrowers taking larger loans may face greater repayment burdens, increasing the likelihood of default | Visualize distribution of `loan_amnt` by `loan_status`, conduct statistical test to confirm difference |
| **H2:** Lower `person_income` is associated with higher default risk | Borrowers with lower income may have limited financial capacity to meet repayment obligations | Visualize distribution of `person_income` by `loan_status`, conduct statistical test to confirm difference |
| **H3:** Lower `loan_grade` (credit quality) is associated with higher default risk | A lower loan grade reflects weaker creditworthiness and higher assessed lending risk | Analyze frequency of defaults across `loan_grade` categories, perform Chi-square test for association |
| **H4:** Shorter `person_emp_length` (employment length) is associated with higher default risk | Borrowers with shorter employment histories may experience less income stability, increasing repayment risk | Visualize distribution of `person_emp_length` by `loan_status`, conduct statistical test to confirm difference |

These hypotheses will be tested through exploratory data analysis and statistical testing to identify whether the respective features are influential predictors of default risk.

## The rationale to map the business requirements to the Data Visualizations and ML tasks
This section explains how each business requirement is addressed by specific analyses, visualizations and ML techniques. It ensures that insights and predictions directly support the business goals and can be interpreted by stakeholders.

**Business Requirement 1: Data Insights (Conventional Analysis)**
- Identify key borrower and loan attributes that are most correlated with loan default.
- Provide visual and statistical insights to help business analysts understand the primary drivers of credit risk.
- Visualize distributions and relationships between key features and the target variable.

**Business Requirement 2: Classification Model (Machine Learning)**
- Develop a **binary classification model** to predict whether a loan applicant is likely to default.
- Show the **probability of default** to support the credit team in decision-making.
- Evaluate model performance and feature importance for transparency and reliability.

**Business Requirement 3: Clustering Model (Machine Learning)**
- Group borrowers into risk-based clusters to segment borrowers by credit behavior and improve tailored intervention strategies.
- Analyze and visualize cluster characteristics to understand risk profiles.
- Visualize cluster assignments to facilitate understanding by stakeholders.


## ML Business Case

#### **Binary Classification Model — Loan Default Prediction**

We aim to develop a **supervised** machine learning model that predicts whether a loan applicant will default or not.  
The model should provide a **probability of default** to support the credit risk team in decision-making.

- **Goal:** Predict if a borrower will default on their loan (`loan_status`) and provide the associated probability of default.  
- **Model type:** Supervised - Binary Classification.  
- **Input features:** Borrower demographic and financial attributes 
- **Model choice:** After experimentation, a **Random Forest** model was chosen as the best-performing and most interpretable model.  
- **Success metrics (on both training and test sets):**  
  - Recall for default ≥ 0.75 – to minimize false negatives (high-risk borrowers predicted as safe)  
  - F1 score for default ≥ 0.60 – ensures a balance between recall and precision  
- **Failure conditions:**  
  - Strong degradation of performance on test data vs. train data → indicates overfitting.  
  - Large imbalance between precision and recall → predictions may not be reliable for business decisions.  
- **Output definition:**  
  - Binary prediction (`0` = no default, `1` = default).  
  - Probability of default (e.g., 0.76 = 76% chance of default) to guide credit risk decisions.  
- **Heuristics:** The model should be used to prioritize risk review and support decision-making (not to fully automate rejections). Thresholds for action should be determined in consultation with the credit risk team to balance loss prevention and borrower impact.
 
#### **Clustering Model — Borrower Segmentation**

We implemented an **unsupervised** clustering model to group borrowers with similar credit and loan characteristics.  
This segmentation helps the credit and retention teams tailor communication, product offerings, and risk mitigation strategies.

- **Goal:** Identify distinct borrower segments based on credit behavior and financial characteristics.  
- **Model type:** Unsupervised - Clustering.  
- **Input features:** Borrower demographic and financial attributes 
- **Model choice:** **K-Means**
- **Success metrics:**  
  - Average silhouette score ≥ 0.45  
  - Clusters are interpretable and distinct in profile characteristics.  
- **Failure conditions:**  
  - Model suggests more than 15 clusters → difficult to interpret or apply in business context.  
  - Clusters are not meaningfully distinct (overlapping feature distributions).  
- **Output definition:**  
  - Cluster assignments appended to the dataset as an additional categorical column (`Clusters`).  
  - Each borrower belongs to one cluster.  
  - Identified cluster characteristics:  
    - **Cluster 0:** Borrowers with a history of previous defaults, mostly renters, moderate income, highest default rate (high-risk).  
    - **Cluster 1:** Borrowers with no history of previous defaults, who mostly rent, lower to mid-range incomes, moderate default rates (middle-risk).  
    - **Cluster 2:** Borrowers with no history of previous defaults, who primarily have mortgages, higher incomes, rarely default (low-risk).  
- **Heuristics:** This clustering provides a systematic segmentation where previously none existed. The results can inform targeted risk interventions and product offerings.


## User Stories
User stories were developed to clearly define the needs and goals of different stakeholders, ensuring that the project dashboard delivers actionable insights and functionality aligned with both business and technical objectives.

1. As a non-technical stakeholder, I want to view a concise and structured overview of the project, including its goals, dataset, and business requirements, so that I can understand what the project aims to achieve and how to navigate the dashboard.

2. As a data analyst, I want to explore correlations and key drivers of loan default through interactive data exploration and visualizations, so that I can identify which borrower and loan attributes most influence default risk and provide data-driven insights to the business.

3. As a business analyst, I want to review the project’s main hypotheses about borrower behavior and validate them with visual and statistical evidence, so that I can understand which factors are meaningfully linked to default and ensure the findings are grounded in data.

4. As a loan officer, I want to input borrower information and receive a predicted probability of default along with a borrower cluster assignment, so that I can make informed lending decisions and take appropriate risk mitigation actions based on the borrower’s profile.

5. As a technical reviewer, I want to examine the predictive model’s structure, key features, and performance metrics, so that I can assess whether the model meets business requirements and delivers reliable and interpretable predictions.

6. As a technical reviewer, I can view borrower clustering insights to evaluate the clustering model’s performance, understand cluster characteristics, and assess how effectively the clusters segment borrowers by default risk.


## Dashboard Design

The dashboard will be developed in **Streamlit** and designed to guide the user from business understanding to actionable insights and model-based predictions.  
It will consist of **six main pages**, each mapped to specific business requirements.

The goal of the dashboard is to provide both **descriptive insights** and **predictive intelligence** to support data-driven decisions in **loan management**.  
It will serve two main user groups:  
- **Business analysts:** who need to explore patterns and trends in borrower data.  
- **Credit officers:** who need actionable information on loan risk and applicant default probability.

### **Page 1: Project Summary**
- **Purpose:** Provide a clear overview of the project and orient users.  
- **Sections:**
  - Project introduction  
  - Project terms & jargon
  - Dataset overview
  - Checkbox: Data inspection (number of rows, columns, and first 10 rows)  
  - Business requirements  
  - Navigation guide for subsequent pages 

### **Page 2: Loan Default Study**
- **Purpose:** Address **Business Requirement 1 (Data Insights)**. This page helps financial institutions understand what drives **default risk**. It focuses on identifying key borrower and loan attributes most correlated with default and provides **visual and statistical insights** for business analysts.
- **Sections:**
  - Correlation Analysis:
    - Checkbox: Display PPS Heatmap to detect both linear and non-linear relationships with the target variable
    - Table of most important features according to PPS score
  - Visualization of main drivers of default 
    - Checkbox: Display distributions of selected key features
    - Summary insights highlight trends
    - Checkbox: Display Parallel Plot to show interactions between multiple key features and their influence on default probability.

### **Page 3: Project Hypotheses and Validation**
- **Purpose:** Present hypotheses and their validation process.  
- **Sections:**
  - State each of the four project hypotheses.  
  - Show validation result:
    - Short written conclusion summarizing whether the hypothesis was confirmed or not
    - Checkbox: Display corresponding distribution plot for each hypothesis and result of statistical test 

### **Page 4: Default Prediction Tool**
- **Purpose:** Address **Business Requirement 2 (Classification Model)** and **Business Requirement 3 (Clustering Model)**
- **Sections:**
  - State Business Requirements 2 and 3 
  - Widget input fields for necessary borrower data  
  - “Run Predictive Analysis” button to send input data through the trained ML pipelines  
  - Output:  
    - Predicted default probability
    - Cluster assignment for additional context  
    - Cluster profile summary 
    - Combined business recommendation based on default probability and cluster  

### **Page 5: Classification Model Insights**
- **Purpose:** Address **Business Requirement 2 (Classification Model)**. Show predictive model performance and interpretation. 
- **Sections:**
  - Describe model objective
  - Overview of used ML pipelines
  - Visualization of the top features contributing to the model’s predictions
  - Insights into model performance
    - Confusion matrix and classification report for both train and test sets 
    - Performance metrics interpretation and conclusions for business relevance

### **Page 6: Borrower Clustering Insights**
- **Purpose:** Address **Business Requirement 3 (Clustering Model)**. Show cluster analysis performance and interpretation. 
- **Sections:**
  - Describe model objective
  - Overview of used ML pipeline
  - Insights into cluster analysis performance
    - Silhouette plot, average silhouette score and number of clusters chosen 
  - Cluster distribution across default levels 
  - Visualization of the top features defining the clusters
  - Description of cluster profiles and business use of segmentation


## Technologies Used

The technologies used throughout the development are listed below.

### Languages

* [Python](https://www.python.org/)

### Python Packages

#### Main Data Analysis & Machine Learning Libraries
* [Pandas](https://pandas.pydata.org/docs/index.html) - Open source library for data manipulation and analysis.
* [NumPy](https://numpy.org/doc/stable/) - Adds support for large, multi-dimensional arrays and high-level mathematical functions.
* [Matplotlib](https://matplotlib.org/) - Comprehensive library for creating static, animated, and interactive visualisations.
* [Seaborn](https://seaborn.pydata.org/) - Statistical data visualisation library for attractive and informative graphics.
* [Plotly Express](https://plotly.com/python/plotly-express/) - High-level library for creating interactive plots easily.
* [scikit-learn](https://scikit-learn.org/stable/) - Open source machine learning library featuring classification, regression, and clustering.
* [Feature-engine](https://feature-engine.trainindata.com/en/latest/) - Library with multiple transformers to engineer and select features for ML models.
* [ppscore](https://pypi.org/project/ppscore/) - Library for detecting linear or non-linear relationships between two features.
* [SciPy](https://scipy.org/) - Library for scientific computing, including statistical tests.
* [XGBoost](https://xgboost.readthedocs.io/en/stable/) - High-performance gradient boosting library.
* [CatBoost](https://catboost.ai/) - Gradient boosting library optimized for categorical features.
* [imbalanced-learn](https://imbalanced-learn.org/stable/) - Tools for handling imbalanced datasets in classification tasks.
* [Yellowbrick](https://www.scikit-yb.org/en/latest/) - Visual analysis and diagnostic tools for machine learning models.
* [Joblib](https://joblib.readthedocs.io/en/stable/) - Utilities for pipelining, caching, and saving/loading models.
* [YData Profiling](https://docs.profiling.ydata.ai/latest/) - Automated exploratory data analysis and profiling reports.

#### Utilities
* [Warnings](https://docs.python.org/3/library/warnings.html) - Suppress or manage warnings in Python.
* [Streamlit](https://docs.streamlit.io/) - Framework for building interactive dashboards and web applications.
* [Glob](https://docs.python.org/3/library/glob.html) - File pathname pattern matching.
* [Zipfile](https://docs.python.org/3/library/zipfile.html) - Work with ZIP archives in Python.

### Other Technologies

* [Git](https://git-scm.com/) - For version control
* [GitHub](https://github.com/) - Code repository
* [Heroku](https://heroku.com) - For application deployment
* [VSCode](https://code.visualstudio.com/) - IDE used for development


## Testing

### User Story Testing
Each page of the dashboard corresponds to a specific user story, ensuring that the application meets the needs of both non-technical and technical users. The dashboard was manually tested using these user stories as a basis for determining success.  

For the Jupyter notebooks, manual testing against user stories was deemed irrelevant, as their execution relies on consecutive functions being successful. Instead, correctness was ensured through code validation and sequential function execution.

#### User Story 1: Project Summary Page
*As a non-technical stakeholder, I want to view a concise and structured overview of the project, including its goals, dataset, and business requirements, so that I can understand what the project aims to achieve and how to navigate the dashboard.*

| Feature | Action | Expected Result | Test Result |
| --- | --- | --- | --- |
| Project Summary Page | Navigate to summary page | Page is displayed with all sections visible; user can read overview, terms, dataset, and business requirements. | Pass |

---

#### User Story 2: Loan Default Study Page
*As a data analyst, I want to explore correlations and key drivers of loan default through interactive data exploration and visualizations, so that I can identify which borrower and loan attributes most influence default risk and provide data-driven insights to the business.*

| Feature | Action | Expected Result | Test Result |
| --- | --- | --- | --- |
| Loan Default Study Page | Navigate to page | Page loads correctly; correlation and feature distributions sections are displayed. | Pass |
| Data Inspection | Tick checkbox to inspect dataset | Table shows number of rows, columns, and first 10 rows. | Pass |
| PPS Heatmap | Tick checkbox to show heatmap | PPS heatmap is displayed with relevant scores. | Pass |
| Feature Distributions | Tick checkbox to show distributions | Visualizations for key features are displayed by default level. | Pass |
| Parallel Plot | Tick checkbox to show parallel plot | Parallel plot displays interactions between features and default probability. | Pass |

---

#### User Story 3: Project Hypotheses Page
*As a business analyst, I want to review the project’s main hypotheses about borrower behavior and validate them with visual and statistical evidence, so that I can understand which factors are meaningfully linked to default and ensure the findings are grounded in data.*

| Feature | Action | Expected Result | Test Result |
| --- | --- | --- | --- |
| Project Hypotheses Page | Navigate to page | Page displays all four hypotheses with introductory text. | Pass |
| Hypothesis Validation | Tick checkbox for each hypothesis | Corresponding distribution plot is displayed along with statistical test result. | Pass |
| Validation Summary | Read page | Written conclusion confirms whether each hypothesis is supported by data. | Pass |

---

#### User Story 4: Default Prediction Tool Page
*As a loan officer, I want to input borrower information and receive a predicted probability of default along with a borrower cluster assignment, so that I can make informed lending decisions and take appropriate risk mitigation actions based on the borrower’s risk profile.*

| Feature | Action | Expected Result | Test Result |
| --- | --- | --- | --- |
| Default Prediction Tool Page | Navigate to page | Page displays explanation of BR2 & BR3, and input widgets are visible. | Pass |
| Input Widgets | Enter unseen borrower data | Widgets respond to input correctly. | Pass |
| Run Predictive Analysis | Click “Run Predictive Analysis” | Predicted default probability, cluster assignment, cluster profile, and business recommendation are displayed. | Pass |

---

#### User Story 5: Classification Model Insights Page
*As a technical reviewer, I want to examine the predictive model’s structure, key features, and performance metrics, so that I can assess whether the model meets business requirements and delivers reliable and interpretable predictions.*

| Feature | Action | Expected Result | Test Result |
| --- | --- | --- | --- |
| Classification Model Insights Page | Navigate to page | Page loads with model overview, ML pipelines, and feature importance visualizations. | Pass |
| Model Metrics | View confusion matrices and classification reports | Train and test set performance metrics are displayed; insights are interpretable by a technical reviewer. | Pass |

---

#### User Story 6: Borrower Clustering Insights Page
*As a technical reviewer, I can view borrower clustering insights to evaluate the clustering model’s performance, understand cluster characteristics, and assess how effectively the clusters segment borrowers by default risk.*

| Feature | Action | Expected Result | Test Result |
| --- | --- | --- | --- |
| Borrower Clustering Insights Page | Navigate to page | Page loads with cluster analysis overview, ML pipeline description, and top feature visualizations. | Pass |
| Cluster Performance Metrics | View silhouette plot and average score | Silhouette plot and score are displayed; number of clusters is indicated. | Pass |
| Cluster Distribution | Analyze default levels by cluster | Correct distribution of borrowers across clusters is shown. | Pass |
| Cluster Profiles | Examine cluster characteristics | Cluster descriptions and business relevance are displayed; technical reviewer can interpret segmentation. | Pass |

### Code Validation
All python code within the `app_pages`and `src` directories as well as the `app.py` file has been validated for PEP8 compliance using Code Institute’s [PEP8 Linter](https://pep8ci.herokuapp.com/). No issues remain.

All code within the `.ipynb` files in `jupyter_notebooks` directory has primarily been checked not to exceed 79 characters per line. Some whitespaces remain.


## Unfixed Bugs
To this date, no known unfixed errors remain in the application, though, even after thorough testing, I cannot rule out the possibility.


## Deployment
### Heroku

The App live link is: https://loan-guard-c4aee35f5523.herokuapp.com/ 

The project was deployed to Heroku using the following steps:

1. Within your working directory, ensure there is a ``setup.sh`` file containing the following:
```
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```
2. Within your working directory, ensure there is a ``.python-version`` file containing a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack supported version of Python.
```
3.12
```
3. Within your working directory, ensure there is a ``Procfile`` file containing the following:
```
web: sh setup.sh && streamlit run app.py
```
4. Ensure your ``requirements.txt`` file contains all the packages necessary to run the streamlit dashboard.
5. Ensure your ``.slugignore`` file contains all  files/directories that are unnecessary for deployment.
```
*.ipynb
```
6. Log in to [Heroku](https://id.heroku.com/login) or create an account if you do not already have one.
7. Click the **New** button on the dashboard and from the dropdown menu select "Create new app".
8. Enter a suitable app name and select your region, then click the **Create app** button.
9. Once the app has been created, navigate to the Deploy tab.
10. At the Deploy tab, in the Deployment method section select **GitHub**.
11. Enter your repository name and click **Search**. Once it is found, click **Connect**.
12. Navigate to the bottom of the Deploy page to the Manual deploy section and select main from the branch dropdown menu.
13. Click the **Deploy Branch** button to begin deployment.
14. The deployment process should happen smoothly if all deployment files are fully functional. Click the button **Open App** at the top of the page to access your App.
15. If the build fails, check the build log carefully to troubleshoot what went wrong. If the slug size is too large then add large files not required for the app to the ``.slugignore`` file.


## Forking and Cloning
If you wish to fork or clone this repository, please follow the instructions below:

### Forking
1. In the top right of the main repository page, click the **Fork** button.
2. Under **Owner**, select the desired owner from the dropdown menu.
3. **OPTIONAL:** Change the default name of the repository in order to distinguish it.
4. **OPTIONAL:** In the **Description** field, enter a description for the forked repository.
5. Ensure the 'Copy the main branch only' checkbox is selected.
6. Click the **Create fork** button.

### Cloning
1. On the main repository page, click the **Code** button.
2. Copy the HTTPS URL from the resulting dropdown menu.
3. In your IDE terminal, navigate to the directory you want the cloned repository to be created.
4. In your IDE terminal, type ```git clone``` and paste the copied URL.
5. Hit Enter to create the cloned repository.


## Credits 

This project has been based on the methodologies used in the [Churnometer](https://github.com/Code-Institute-Solutions/churnometer) project 
from Code Institute. Some functions from that project have been used in their 
original form, while others have been customized for the purposes of this project. 
The main functions that have been used are:

* Functions for calculating correlations and displaying correlation heatmaps.  
* Customized function for testing different numerical transformations to evaluate 
  improvement in distribution shape.  
* Function for performing hyperparameter optimization using grid search, adapted 
  to also perform randomized search.  
* Function to display confusion matrix and classification performance report.  
* Cluster analysis utilities:  
  - Visualization of PCA results, elbow and silhouette plots.  
  - Cluster distribution plots.  
  - Function to describe cluster profiles.
