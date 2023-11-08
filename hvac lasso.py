#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import Lasso
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


# In[2]:


# Load the data
df = pd.read_excel('data_HVAC.xlsx')
y = df['HVAC']

# Perform ACF and PACF analysis
plot_acf(df, lags=50)
plot_pacf(df, lags=50)


# In[3]:


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


# In[4]:


from statsmodels.tsa.stattools import pacf
# Create X and y variables
X = create_lag_features(df)
y = df.copy().iloc[:, 0]
X = X.iloc[48:, :]
y = y.iloc[48:]


# In[5]:


#Split the data into train and test sets
train_size = int(len(X) * 0.7)
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]


# In[6]:


# Define the search space for hyperparameters
search_space = {
    'alpha': Real(0.01, 10, prior='log-uniform'),
    'tol': Real(1e-13, 1e-2, prior='log-uniform')
}

# Create a Lasso regression model
model = Lasso()


# In[7]:


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


# In[8]:


# Print the best hyperparameters and corresponding mean score
print("Best hyperparameters: ", bayes_cv.best_params_)
print("Best mean score: ", bayes_cv.best_score_)


# In[9]:


# Predict on the test set
y_pred = bayes_cv.predict(test_X)

# Calculate and print the RMSE, MAE, and R-squared score on the test set
print("RMSE on test set: ", mean_squared_error(test_y, y_pred, squared=False))
print("MAE on test set: ", mean_absolute_error(test_y, y_pred))
print("R-squared on test set: ", r2_score(test_y, y_pred))


# In[10]:


import os
import pandas as pd

# Predict on test set
y_pred = bayes_cv.predict(test_X)

# Create dataframe with actual and predicted values
df_results = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

# Create a filename for the Excel file
filename = 'actual_vs_predicted100.xlsx'

# Get the path to the Downloads folder
downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')

# Create the full file path
file_path = os.path.join(downloads_path, filename)

# Save the dataframe to Excel
df_results.to_excel(file_path, index=False)

print(f'Excel file saved to {file_path}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




