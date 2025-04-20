#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[9]:


cybersec_df = pd.read_csv("./Data/CySecData.csv") 


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[10]:


cybersec_df.head()


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[11]:


cybersec_df.info()
cybersec_df.describe()


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[7]:


categorical_cols = cybersec_df.select_dtypes(include='object').columns.drop('class')
dfDummies = pd.get_dummies(cybersec_df, columns=categorical_cols)


# In[14]:


numerical_cols = cybersec_df.drop(columns=categorical_cols)
dfDummies = pd.concat([numerical_cols, dfDummies], axis=1)


# In[15]:


dfDummies.head()


# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[18]:


dfDummies = dfDummies.drop('class', axis=1)


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[19]:


from sklearn.preprocessing import StandardScaler


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[20]:


scaler = StandardScaler()
dfNormalized = scaler.fit_transform(dfDummies)


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[21]:


X = dfDummies
y = cybersec_df['class']


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[24]:


from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.model_selection import train_test_split  # Splitting the dataset
from sklearn.model_selection import cross_val_score, KFold  # Cross-validation


# # Step 11: Defining the models (Logistic Regression, Support Vector Machine, Random Forest)
# TODO: Define the models to be evaluated.

# In[23]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models


# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.
# Hint: Use Kfold cross validation and a loop

# In[25]:


results = []
names = []

kfold = KFold(n_splits=10, random_state=42, shuffle=True)

for name, model in models:
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: Mean Accuracy = {cv_results.mean():.4f}, Std = {cv_results.std():.4f}")


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[ ]:


#get_ipython().system('jupyter nbconvert --to python notebook.ipynb')

