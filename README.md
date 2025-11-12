# Loan Guard

Developed by [kathrinmzl](https://www.github.com/kathrinmzl)

[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/kathrinmzl/LoanGuard)](https://www.github.com/kathrinmzl/LoanGuard/commits/main)
![GitHub last commit](https://img.shields.io/github/last-commit/kathrinmzl/LoanGuard?color=red)
![GitHub language count](https://img.shields.io/github/languages/count/kathrinmzl/LoanGuard?color=yellow)
![GitHub top language](https://img.shields.io/github/languages/top/kathrinmzl/LoanGuard?color=green)
[![badge](https://img.shields.io/badge/deployment-Heroku-purple)](https://loan-guard-c4aee35f5523.herokuapp.com/)

In the banking sector, effective credit risk assessment is critical to maintaining financial stability and minimizing losses. Loan defaults can lead to significant financial setbacks and reduced liquidity for lending institutions. The LoanGuard project aims to help financial institutions better understand what drives default risk and proactively identify borrowers who are likely to default on their loans.


The project was created for educational purposes only.

[Live page on Heroku](https://loan-guard-c4aee35f5523.herokuapp.com/)

![Application Mockup](docs/mockup.png) ???

Source: [amiresponsive](https://ui.dev/amiresponsive?url=  TODO LINK!!) ???


## Dataset Content

The used dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) and contains information about individual borrowers and their loan characteristics. Each row represents a loan record, including both personal and financial attributes that may influence the likelihood of default. The dataset provides a comprehensive overview of borrower profiles, such as age, income, home ownership or employment details, as well as loan-specific features like loan amount, interest rate and purpose.  

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
| `loan_status` | Loan default flag (target variable) | Target | int64 | 0 = No Default, 1 = Default |


## Project Terms & Jargon 
- A **borrower** is a person who takes out a loan from a financial institution.  
- A **loan** is an amount of money borrowed that is expected to be paid back with interest.  
- A **default** occurs when a borrower fails to make scheduled loan payments or meet the agreed repayment terms.  
- A **defaulted borrower** is a borrower who has failed to repay their loan as agreed and is classified as being in default.  
- A **non-default** refers to a borrower who repays their loan successfully or continues to make payments on time.  


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

**Business Requirement 2: Predictive Model (Machine Learning)**
- Develop a machine learning model capable of predicting whether a loan applicant is likely to default. The system should output a probability of default to support the credit team in decision-making.

**Business Requirement 3: Clustering Model (Machine Learning)** - Optional
- Group borrowers into risk-based clusters to segment borrowers by credit behavior and improve tailored intervention strategies.

Additional Considerations: TODO -Adjust/take out?

- Predictions should be explainable, highlighting which factors most influence a borrower’s default risk. (OPTIONAL: SHAP/LIME)

- Ethical and fairness considerations must be taken into account to avoid bias against demographic groups. (OPTIONAL: Check later what could done here)

- The final output will be presented in an interactive dashboard that allows real-time testing of new loan applications.


## Hypotheses and how to validate them?
To better understand the factors influencing loan default risk, we formulated four key hypotheses based on domain knowledge and the available data. Each hypothesis focuses on a variable expected to impact default probability.

| Hypothesis | Rationale | Validation |
|------------|------------|-------------|
| **H1:** Higher `loan_amnt` is associated with higher default risk | Borrowers taking larger loans may face greater repayment burdens, increasing the likelihood of default | Visualize distribution of `loan_amnt` by `loan_status`, conduct statistical test to confirm difference, correlation analysis, correlation analysis |
| **H2:** Lower `person_income` is associated with higher default risk | Borrowers with lower income may have limited financial capacity to meet repayment obligations | Visualize distribution of `person_income` by `loan_status`, conduct statistical test to confirm difference, correlation analysis, correlation analysis |
| **H3:** Lower `loan_grade` (credit quality) is associated with higher default risk | A lower loan grade reflects weaker creditworthiness and higher assessed lending risk | Analyze frequency of defaults across `loan_grade` categories, perform Chi-square test for association |
| **H4:** Shorter `person_emp_length` (employment length) is associated with higher default risk | Borrowers with shorter employment histories may experience less income stability, increasing repayment risk | Visualize distribution of `person_emp_length` by `loan_status`, conduct statistical test to confirm difference, correlation analysis, correlation analysis |

These hypotheses will be tested through exploratory data analysis and modeling to identify the most influential predictors of default risk.

## The rationale to map the business requirements to the Data Visualizations and ML tasks
This section explains how each business requirement is addressed by specific analyses, visualizations and ML techniques. It ensures that insights and predictions directly support the business goals and can be interpreted by stakeholders.

**Business Requirement 1: Data Insights (Conventional Analysis)**
- Identify key borrower and loan attributes that are most correlated with loan default.
- Provide visual and statistical insights to help business analysts understand the primary drivers of credit risk.
- Visualize distributions and relationships between key features and the target variable.

**Business Requirement 2: Predictive Model (Machine Learning)**
- Develop a **binary classification model** to predict whether a loan applicant is likely to default.
- Additionally, show the **probability of default** to support the credit team in decision-making.
- Evaluate model performance and feature importance for transparency and reliability.

**Business Requirement 3: Clustering Model (Machine Learning) - Optional**
- Group borrowers into risk-based clusters to segment borrowers by credit behavior and improve tailored intervention strategies.
- Analyze and visualize cluster characteristics to understand risk profiles.
- Visualize cluster assignments to facilitate understanding by stakeholders. (OPTIONAL)


## ML Business Case

#### **Binary Classification Model — Loan Default Prediction**

We aim to develop a **supervised** machine learning model that predicts whether a loan applicant  will default or not.  
In addition to the binary outcome, the model will provide a **probability of default** to support the credit risk team in decision-making.

- **Goal:** Predict if a borrower will default on their loan (`loan_status`) and provide the associated probability of default.  
- **Model type:** Supervised — Binary Classification.  
- **Input features:** All borrower demographic and financial attributes
- **Model choice:** To be determined after experimentation. Candidate models include Logistic Regression, Random Forest, and Gradient Boosting.  
- **Success metrics (on both training and test sets):**  
  - Recall for default ≥ 0.75 – to minimize false negatives (high-risk borrowers predicted as safe)
  - F1 score ≥ 0.60 – ensures a balance between recall and precision
- **Failure conditions:**  
  - Strong degradation of performance on test data vs. train data → indicates overfitting.  
  - Large imbalance between precision and recall → unreliable predictions.  
- **Output definition:**  
  - Binary prediction (`0` = no default, `1` = default).   
  - Probability of default (e.g., 0.76 = 76% chance of default) to guide credit risk decisions.  
- **Heuristics:** Traditionally, financial institutions rely on fixed credit scores or manual reviews to assess loan risk. The model should be used to prioritize risk review and support decision-making (not to fully automate rejections). Thresholds for action should be set in consultation with credit risk stakeholders to balance loss prevention and borrower impact.
 

#### **Clustering Model — Borrower Segmentation (Optional)**

We plan to explore an **unsupervised** clustering model to group borrowers with similar credit and loan characteristics.  
This segmentation will help the credit and retention teams tailor communication, product offerings, and risk mitigation strategies.

- **Goal:** Identify distinct borrower segments based on credit behavior and financial characteristics.  
- **Model type:** Unsupervised — Clustering.  
- **Input features:** Selected normalized numerical and encoded categorical variables that reflect borrower behavior and financial profile.  
- **Model choice:** To be determined after exploration (likely K-Means or Hierarchical Clustering).  
- **Success metrics:**  
  - Average silhouette score ≥ 0.45  
  - Clusters should be interpretable and distinct in profile characteristics.  
- **Failure conditions:**  
  - Model suggests more than 15 clusters → difficult to interpret or apply in business context.  
  - Clusters are not meaningfully distinct (overlapping feature distributions).  
- **Output definition:**  
  - Cluster assignments appended to the dataset as an additional categorical column (`ClusterID`).  
  - Each borrower belongs to one cluster (e.g., 0, 1, 2, …).  
- **Heuristics:** Currently, no formal segmentation process exists.


## Dashboard Design TODO Anpassen je nachdem was wirklich drin ist

The dashboard will be developed in **Streamlit** and designed to guide the user from business understanding to actionable insights and model-based predictions.  
It will consist of **five main pages**, each mapped to specific business requirements.

The goal of the dashboard is to provide both **descriptive insights** and **predictive intelligence** to support data-driven decisions in **loan management and credit risk assessment**.  
It will serve two main user groups:  
- **Business analysts:** who need to explore patterns and trends in borrower data.  
- **Credit officers:** who need actionable information on loan risk and applicant default probability.

### **Page 1: Project Summary**
- **Purpose:** Provide a clear overview of the project and orient users.  
- **Sections:**
  - Project introduction  
  - Project terms & jargon
  - Dataset overview
  - Business requirements  
  - Navigation guide for subsequent pages 

### **Page 2: Loan Default Study**
- **Purpose:** Address **Business Requirement 1 (Data Insights)**  
- **Sections:**
  - Checkbox: Data inspection (number of rows, columns, and first 10 rows)  
  - Correlation heatmap of numerical variables  
  - Visualization of main drivers of default (e.g., boxplots or histograms for `person_income`, `loan_amnt`, `loan_int_rate`, etc.)  
  - Checkbox: Display pairplot or parallel coordinates plot for top correlated variables  
  - Correlation conclusions and considerations  
  - Optional: Display summary statistics and data distribution insights  

### **Page 3: Project Hypotheses and Validation**
- **Purpose:** Present hypotheses and their validation process.  
- **Sections:**
  - State each of the four project hypotheses.  
  - Checkbox: Display corresponding plot for each hypothesis (e.g., boxplot or histogram split by `loan_status`)  
  - Checkbox: Display test results (e.g., t-test or chi-square)  
  - Short written conclusions summarizing which hypotheses were validated  
  - Insights and next steps (how findings inform feature selection or model design)  

### **Page 4: Default Prediction Tool**
- **Purpose:** Address **Business Requirement 2 (Predictive Model)**  
- **Sections:**
  - State Business Requirement 2  
  - Widget input fields for neccessary borrower data  
  - “Run Predictive Analysis” button to send input data through the trained ML pipeline  
  - Output:  
    - Predicted default status (Yes/No)  
    - Probability of default (e.g., 73%)  
    - Top 3 contributing factors (based on feature importance or SHAP) 
  
### **Page 5: Classification Model Insights**
- **Purpose:** Show model performance and interpretation.  
- **Sections:**
  - Model overview and ML pipeline steps  
  - Model evaluation metrics (Precision, Recall, F1-score, Accuracy, AUC)  
  - Confusion matrix, ROC curve, Precision–Recall curve  
  - Feature importance visualization 
  - Considerations and conclusions (model interpretability and limitations)  

### **Page 6: Borrower Clustering Insights (Optional)**
- **Purpose:** Address **Business Requirement 3 (Clustering Model)**  
- **Sections:**
  - Model overview (unsupervised clustering rationale)
  - Silhouette score and number of clusters chosen
  - 2D visualization of clusters (using PCA/t-SNE)
  - Cluster profile table (avg. Income, CreditScore, Default rate, LoanPurpose)
  - Bar chart comparing clusters by default rate
  - Cluster interpretation summary (e.g., “Cluster 2 — low income, high DTI, high default risk”)
  - Considerations and conclusions (business use of segmentation)



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people who provided support through this project.

