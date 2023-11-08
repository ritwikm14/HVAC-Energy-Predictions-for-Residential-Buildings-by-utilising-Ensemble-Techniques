#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


from sklearn.tree import DecisionTreeRegressor


# In[8]:


# Define the search space for hyperparameters
search_space = {
    'max_depth': Integer(2, 30),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': ['auto', 'sqrt', 'log2'],
}

# Create a DecisionTreeRegressor model
model = DecisionTreeRegressor()


# In[9]:


# Define the Bayes Search CV object
bayes_cv = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=1
)


# In[11]:



# Fit the Bayes Search CV object to the training data
bayes_cv.fit(train_X, train_y)


# In[12]:


# Use the best hyperparameters to create a final DecisionTreeRegressor model
model = DecisionTreeRegressor(**bayes_cv.best_params_)
model.fit(train_X, train_y)


# In[13]:


# Evaluate the model on the test data
y_pred = model.predict(test_X)
rmse = np.sqrt(mean_squared_error(test_y, y_pred))
mae = mean_absolute_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)


# In[14]:



print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R^2:", r2)


# In[51]:


import os
import pandas as pd

# Predict on test set
y_pred = bayes_cv.predict(test_X)

# Create dataframe with actual and predicted values
df_results = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

# Create a filename for the Excel file
filename = 'actual_vs_predicted7.xlsx'

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





# In[ ]:





# In[ ]:





# In[ ]:




