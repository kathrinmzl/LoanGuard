# Loan Guard

Developed by [kathrinmzl](https://www.github.com/kathrinmzl)

[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/kathrinmzl/LoanGuard)](https://www.github.com/kathrinmzl/LoanGuard/commits/main)
![GitHub last commit](https://img.shields.io/github/last-commit/kathrinmzl/LoanGuard?color=red)
![GitHub language count](https://img.shields.io/github/languages/count/kathrinmzl/LoanGuard?color=yellow)
![GitHub top language](https://img.shields.io/github/languages/top/kathrinmzl/LoanGuard?color=green)
[![badge](https://img.shields.io/badge/deployment-Heroku-purple)]( TODO LINK!!)

In the banking sector, effective credit risk assessment is critical to maintaining financial stability and minimizing losses. Loan defaults can lead to significant financial setbacks and reduced liquidity for lending institutions. The LoanGuard project aims to help financial institutions better understand what drives default risk and proactively identify customers who are likely to default on their loans.


The project was created for educational purposes only.

[Live page on Heroku]( TODO LINK!!)

![Application Mockup](docs/mockup.png) ???

Source: [amiresponsive](https://ui.dev/amiresponsive?url=  TODO LINK!!) ???


## Dataset Content

The used dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default) and contains information about individual borrowers and their loan characteristics. Each row represents a unique loan record, including both personal and financial attributes that may influence the likelihood of default. The dataset provides a comprehensive overview of borrower profiles, such as age, income, credit score, employment details, and marital status, as well as loan-specific features like loan amount, interest rate, loan term, and purpose.  

In total, the dataset includes **255,347 records and 18 variables**. The target variable, **`Default`**, indicates whether a borrower has defaulted on their loan (`1`) or successfully repaid it (`0`). This data allows for predictive modeling to identify patterns and risk factors associated with loan defaults.  

| Variable          | Meaning                         | Column Type | Data Type | Units / Possible Values |
|------------------|---------------------------------|-------------|-----------|------------------------|
| LoanID           | Unique identifier for each loan | Identifier  | object    | Unique code per row    |
| Age              | Age of the customer             | Feature     | int       | 18 - 69 Years          |
| Income           | Annual income of the customer   | Feature     | float     | 15000.0 - 149999.0 USD |
| LoanAmount       | Amount of the loan              | Feature     | float     | 5000.0 - 249999.0 USD |
| CreditScore      | Credit score of the customer    | Feature     | float     | 300 - 849              |
| MonthsEmployed   | Number of months employed       | Feature     | int       | 0 - 119 Months         |
| NumCreditLines   | Number of open credit lines     | Feature     | int       | 1 - 4                  |
| InterestRate     | Interest rate of the loan       | Feature     | float     | 2 - 25 %               |
| LoanTerm         | Loan term in months             | Feature     | int       | 12 - 60 Months         |
| DTIRatio         | Debt-to-income ratio            | Feature     | float     | 0.1 - 0.9 Ratio        |
| Education        | Highest education level         | Feature     | object    | Bachelor's, Master's, High School, PhD |
| EmploymentType   | Type of employment              | Feature     | object    | Full-time, Unemployed, Self-employed, Part-time |
| MaritalStatus    | Marital status                  | Feature     | object    | Divorced, Married, Single |
| HasMortgage      | Mortgage ownership              | Feature     | object    | Yes, No                |
| HasDependents    | Has dependents                  | Feature     | object    | Yes, No                |
| LoanPurpose      | Purpose of the loan             | Feature     | object    | Other, Auto, Business, Home, Education |
| HasCoSigner      | Has a co-signer                 | Feature     | object    | Yes, No                |
| Default          | Loan default flag               | Target      | int       | 0 = No default, 1 = Default |

## Project Terms & Jargon TODO
	- A customer is a person who consumes your service or product.
	- A prospect is a potential customer.
	- A churned customer is a user who has stopped using your product or service.
	- This customer has a tenure level, which is the number of months this person has used our product/service.

## Business Requirements
From a business perspective, this project supports the strategic goals of a financial institution such as:

- Improving risk management by identifying high-risk applicants early.

 - Enhancing profitability through optimized loan approval decisions.

- Increasing customer trust and operational efficiency by offering fair, data-driven credit evaluations.

 - Enabling personalized loan offerings and proactive interventions for at-risk customers (e.g., adjusted payment plans or counseling).

Ultimately, this project aligns predictive analytics with the bank’s long-term objective of balancing growth with financial stability.

To achieve the outlined objectives, the project will focus on the following key requirements.

**Business Requirement 1: Data Insights (Conventional Analysis)**
- Identify key customer and loan attributes that are most correlated with loan default.
Provide visual and statistical insights to help business analysts understand the primary drivers of credit risk.

**Business Requirement 2: Predictive Model (Machine Learning)**
- Develop a machine learning model capable of predicting whether a loan applicant is likely to default. The system should output a probability of default to support the credit team in decision-making.

**Business Requirement 3: Clustering Model (Machine Learning)** - Optional
- Group borrowers into risk-based clusters to segment customers by credit behavior and improve tailored intervention strategies.

Additional Considerations: TODO -Adjust/take out?

- Predictions should be explainable, highlighting which factors most influence a customer’s default risk. (OPTIONAL: SHAP/LIME)

- Ethical and fairness considerations must be taken into account to avoid bias against demographic groups. (OPTIONAL: Check later what could done here)

- The final output will be presented in an interactive dashboard that allows real-time testing of new loan applications.


## Hypotheses and how to validate them?
To better understand the factors influencing loan default risk, we formulated four key hypotheses based on domain knowledge and the available data. Each hypothesis focuses on a variable expected to impact default probability.

| Hypothesis | Rationale | Validation |
|------------|-----------|----------------|
| H1: Higher LoanAmount is associated with higher default risk | Borrowers taking larger loans may struggle more to repay | Visualize distribution of LoanAmount by Default, conduct statistical test to confirm difference, correlation analysis |
| H2: Lower CreditScore is associated with higher default risk | Credit score reflects repayment history and creditworthiness | Visualize distribution of CreditScore by Default, conduct statistical test to confirm difference, correlation analysis |
| H3: Shorter MonthsEmployed is associated with higher default risk | Less stable employment may indicate lower income stability | Visualize distribution of MonthsEmployed by Default, conduct statistical test to confirm difference, correlation analysis |
| H4: Higher DTIRatio is associated with higher default likelihood | High debt relative to income may indicate repayment difficulty | Visualize distribution of DTIRatio by Default, conduct statistical test to confirm difference, correlation analysis |


## The rationale to map the business requirements to the Data Visualizations and ML tasks
This section explains how each business requirement is addressed by specific analyses, visualizations and ML techniques. It ensures that insights and predictions directly support the business goals and can be interpreted by stakeholders.

**Business Requirement 1: Data Insights (Conventional Analysis)**
- Identify key customer and loan attributes that are most correlated with loan default.
- Provide visual and statistical insights to help business analysts understand the primary drivers of credit risk.
- Visualize distributions and relationships between key features and the target variable.

**Business Requirement 2: Predictive Model (Machine Learning)**
- Develop a **binary classification model** to predict whether a loan applicant is likely to default.
- Additionally, show the **probability of default** to support the credit team in decision-making.
- Evaluate model performance and feature importance for transparency and reliability.

**Business Requirement 3: Clustering Model (Machine Learning) - Optional**
- Group borrowers into risk-based clusters to segment customers by credit behavior and improve tailored intervention strategies.
- Analyze and visualize cluster characteristics to understand risk profiles.
- Visualize cluster assignments to facilitate understanding by stakeholders. (OPTIONAL)


## ML Business Case
* In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



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

