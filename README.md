# Telco_Classification_Project
-Welcome to my first machine learning project! 

## Project Objectives
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
- Create modules (acquire.py, prepare.py) that make your process repeateable.
- Construct a model to predict customer churn using classification techniques.
- Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using your Jupyter Notebook from above; your presentation should be appropriate for your target audience.
- Answer panel questions about your code, process, findings and key takeaways, and model.

## Goals
- Find drivers for customer churn at Telco. Why are customers churning?
- Construct a ML classification model that accurately predicts customer churn.
- Document your process well enough to be presented or read like a report.

## Audience

Target audience for my notebook walkthrough is the Codeup Data Science team. 



## Deliverables
1.) a Jupyter Notebook Report showing process and analysis with the goal of finding drivers for customer churn. This notebook should be commented and documented well enough to be read like a report or walked through as a presentation.

2.) a README.md file containing the project description with goals, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), key findings, recommendations, and takeaways from your project.

3.) a CSV file with customer_id, probability of churn, and prediction of churn. (1=churn, 0=not_churn). These predictions should be from your best performing model ran on X_test. Note that the order of the y_pred and y_proba are numpy arrays coming from running the model on X_test. The order of those values will match the order of the rows in X_test, so you can obtain the customer_id from X_test and concatenate these values together into a dataframe to write to CSV.

4.) individual modules, .py files, that hold your functions to acquire and prepare your data.

5.) a notebook walkthrough presentation with a high-level overview of your project (5 minutes max). You should be prepared to answer follow-up questions about your code, process, tests, model, and findings.


### Data Dictionary



Feature	Data Type	Description
customer_id	object	unique customer ID
payment_type_id	int64	Type of payment: credit card(automatic), Bank transfer(automatic), Mailed check
partner	object	0= no partner, 1= has partner
phone_service	object	Yes or No for phone service
tenure	int64	# of months with company
total_charges	int64	total charges since day 1
churn	object	Yes = Churn, No = Not Churned
churn_No	object	0 = Churn, 1 = Not Churned
churn_Yes	object	1 = Churn, 0 = Not Churned
no_partner_depend	int64	no partner & no dependents
phone_lines	int64	1 = has phone lines, 0 = No phone
stream_tv_mov	int64	has streaming tv & streaming movie
online_sec_bckup	int64	has online security & online backup
female	uint8	1 = female, 0 = not female
male	uint8	1 = male, 0 = not male
no_partner	uint8	1 = no partner, 0 = has partner
has_partner	unit8	1 = has partner, 0 = no partner
dependents_no	unit8	1 = no dependents, 0 = has dependents
dependents_yes	unit8	1 = has dependents, 0 = no dependents
device_proctection_no	uint8	1 = no protection, 0 = has protection
device_proctection_no_int	uint8	1 = no internet, 0 = has internet
device_proctection_yes	uint8	1 = has protection, 0 = no protection
tech_support_No	uint8	1 = no tech support, 0 = has tech support
tech_support_No internet service	uint8	1 = no internet, 0 = has internet
tech_support_yes	uint8	1 = has tech support, 0 = no tech support
paperless_billing_no	uint8	1 = no paperless billing 0 = has paperless billing
paperless_billing_yes	uint8	1 = has paperless billing, 0 = no paperless billing
contract_type_Month-to-month	uint8	1 = on monthly contract, 0 = no monthly contract
contract_type_One year	uint8	1 = on 1 yr contract, 0 = not on 1 yr contract
contract_type_Two year	uint8	1 = on 2 yr contract, 0 = not on 2 yr contract
internet_service_type_DSL	uint8	1 = has dsl, 0 = no dsl
internet_service_type_Fiber optic	uint8	1 = has fiber optic, 0 = no fiber optic
internet_service_type_None	uint8	1 = no internet, 0 = has internet
payment_type_Bank transfer (automatic)	uint8	1 = pay w/bank transfer, 0 = no bank transfer
payment_type_Credit card (automatic)	uint8	1 = pays w/credit card, 0 = no credit card
payment_type_Electronic check	uint8	1 = pays w/elec check, 0 = no elec check
payment_type_Mailed check	uint8	1 = pays w/mail check, 0 = no mail check




## Pipeline Stages Breakdown

### Plan 

-Create README.md with data dictionary, project and business goals, come up with initial hypotheses.

-Acquire data from the Codeup Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.

-Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the funtion.

-Clearly define two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.

-Establish a baseline accuracy and document well.

-Train three different classification models.

-Evaluate models on train and validate datasets.

-Choose the model with that performs the best and evaluate that single model on the test dataset.

-Create csv file with the measurement id, the probability of the target values, and the model's prediction for each observation in my test dataset.

-Document conclusions, takeaways, and next steps in the Final Report Notebook.

### Acquire 

-Store functions that are needed to acquire data from the measures and species tables from the telco_churn database on the Codeup data science database server; make sure the acquire.py module contains the necessary imports to run my code.

-The final function will return a pandas DataFrame.

-Import the acquire function from the acquire.py module and use it to acquire the data in the Final Report Notebook.

-Complete some initial data summarization (.info(), .describe(), .value_counts(), ...).

Plot distributions of individual variables.

### Prepare

-Store functions needed to prepare the telco_churn data; make sure the module contains the necessary imports to run the code. 
The final function should do the following: 
Split the data into train/validate/test. 
- Handle any missing values. 
- Handle erroneous data and/or outliers that need addressing. 
- Encode variables as needed. 
- Create any new features, if made for this project.
- Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.

### Explore

-Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, species.

Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.

Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to species (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.

Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.

### Model

-Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document these steps well.

-Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.

-Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.

-Feature Selection (after initial iteration through pipeline): Are there any variables that seem to provide limited to no additional information? If so, remove them.

-Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.

-Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.

### Deliver

-Introduce myself and my project goals at the very beginning of my notebook walkthrough.

-Summarize my findings at the beginning like I would for an Executive Summary. 

-Walk Codeup Data Science Team through the analysis I did to answer my questions and that lead to my findings. 








