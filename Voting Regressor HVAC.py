#!/usr/bin/env python
# coding: utf-8

# In[2]:


from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
get_ipython().system('pip install scikit-optimize')
from skopt import BayesSearchCV

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas.plotting as pdplt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


# In[3]:


# Load the data
df = pd.read_excel('data_HVAC.xlsx')
y = df['HVAC']

# Perform ACF and PACF analysis
plot_acf(df, lags=50)
plot_pacf(df, lags=50)


# In[4]:


import pandas.plotting as pdplt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
# Create lag features
# Create lag features
def create_lag_features(y):
    scaler = StandardScaler()
    df = pd.DataFrame()
    partial = pd.Series(data=pacf(y, nlags=48))
    lags = list(partial[np.abs(partial) >= 0.2].index)
    lags.remove(0)  # Avoid inserting the time series itself
    for l in lags:
        df[f"lag_{l}"] = y.shift(l)
    features = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
    features.index = y.index
    return features


# In[5]:


from statsmodels.tsa.stattools import pacf
# Create X and y variables
X = create_lag_features(df)
y = df.copy().iloc[:, 0]
X = X.iloc[48:, :]
y = y.iloc[48:]


# In[6]:


#Split the data into train and test sets
train_size = int(len(X) * 0.7)
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]


# In[7]:



# Define the search space for hyperparameters
search_space = {
    'learning_rate': Real(0.01, 1, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(100, 1000),
    'reg_lambda': Real(1e-6, 10, prior='log-uniform'),
    'gamma': Real(1e-6, 1, prior='log-uniform')
}

# Create an XGBoost model
model = XGBRegressor(objective='reg:squarederror')


# In[ ]:


# Define the Bayes Search CV object
bayes_cv = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=10,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the Bayes Search CV object to the training data
bayes_cv.fit(train_X, train_y)


# In[9]:


# Print the best hyperparameters and corresponding mean score
print("Best hyperparameters: ", bayes_cv.best_params_)
print("Best mean score: ", bayes_cv.best_score_)


# In[10]:



# Predict on the test set
y_pred = bayes_cv.predict(test_X)


# In[11]:



# Calculate and print the RMSE, MAE, and R-squared score on the test set
print("RMSE on test set: ", mean_squared_error(test_y, y_pred, squared=False))
print("MAE on test set: ", mean_absolute_error(test_y, y_pred))
print("R-squared on test set: ", r2_score(test_y, y_pred))



# In[ ]:


import os
import pandas as pd



# Create dataframe with actual and predicted values
df_results = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

# Create a filename for the Excel file
filename = 'xgboost_hvac.xlsx'

# Get the path to the Downloads folder
downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')

# Create the full file path
file_path = os.path.join(downloads_path, filename)

# Save the dataframe to Excel
df_results.to_excel(file_path, index=False)

print(f'Excel file saved to {file_path}')


# In[13]:


get_ipython().system('pip install catboost')
from catboost import CatBoostRegressor
# Create lag features
def create_lag_features(y):
    scaler = StandardScaler()
    df = pd.DataFrame()
    partial = pd.Series(data=pacf(y, nlags=48))
    lags = list(partial[np.abs(partial) >= 0.2].index)
    lags.remove(0)
    for l in lags:
        df[f"lag_{l}"] = y.shift(l)
    features = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
    features.index = y.index
    return features

# Create X and y variables
X = create_lag_features(df)
y = df.copy().iloc[:, 0]
X = X.iloc[48:, :]
y = y.iloc[48:]


# In[14]:


# Split the data into train and test sets
train_size = int(len(X) * 0.7)
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# Define the search space for hyperparameters (CatBoost specific hyperparameters)
search_space_catboost = {
    'learning_rate': Real(0.01, 1, prior='log-uniform'),
    'depth': Integer(3, 10),
    'iterations': Integer(100, 1000),
    'l2_leaf_reg': Real(1e-6, 10, prior='log-uniform'),
    'custom_metric': ['RMSE']
}

# Create a CatBoost model
model_catboost = CatBoostRegressor()


# In[15]:


# Define the Bayes Search CV object
bayes_cv_catboost = BayesSearchCV(
    estimator=model_catboost,
    search_spaces=search_space_catboost,
    n_iter=10,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the Bayes Search CV object to the training data
bayes_cv_catboost.fit(train_X, train_y)


# In[16]:


# Print the best hyperparameters and corresponding mean score
print("Best hyperparameters: ", bayes_cv_catboost.best_params_)
print("Best mean score: ", bayes_cv_catboost.best_score_)


# In[21]:


# Predict on the test set
y_pred_catboost = bayes_cv_catboost.predict(test_X)


# In[22]:


# Calculate and print the RMSE, MAE, and R-squared score on the test set
print("RMSE on test set (CatBoost): ", mean_squared_error(test_y, y_pred_catboost, squared=False))
print("MAE on test set (CatBoost): ", mean_absolute_error(test_y, y_pred_catboost))
print("R-squared on test set (CatBoost): ", r2_score(test_y, y_pred_catboost))


# In[ ]:


import os
import pandas as pd



# Create dataframe with actual and predicted values
df_results = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

# Create a filename for the Excel file
filename = 'xgboost_hvac.xlsx'

# Get the path to the Downloads folder
downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')

# Create the full file path
file_path = os.path.join(downloads_path, filename)

# Save the dataframe to Excel
df_results.to_excel(file_path, index=False)

print(f'Excel file saved to {file_path}')


# In[23]:


from sklearn.ensemble import VotingRegressor


# In[26]:



# Create Voting Regressor

voting_regressor = VotingRegressor(
    estimators=[('xgb_model', model), ('catboost', model_catboost)]
)


# In[27]:


# Fit the Voting Regressor
voting_regressor.fit(train_X, train_y)


# In[28]:


# Predict on the test set
y_pred_voting = voting_regressor.predict(test_X)


# In[29]:



# Calculate and print the RMSE, MAE, and R-squared score on the test set
print("RMSE on test set (Voting Regressor): ", mean_squared_error(test_y, y_pred_voting, squared=False))
print("MAE on test set (Voting Regressor): ", mean_absolute_error(test_y, y_pred_voting))
print("R-squared on test set (Voting Regressor): ", r2_score(test_y, y_pred_voting))


# In[ ]:





# In[ ]:




