#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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


# In[4]:


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

# Split the data into train and test sets
train_size = int(len(X) * 0.7)
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]


# In[62]:


# Define the model architecture
model = Sequential()
model.add(Dense(64, input_shape=(train_X.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))


# In[63]:



# Compile the model
model.compile(loss='mse', optimizer=Adam(lr=0.000001))


# In[64]:


# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=10)


# In[ ]:


# Train the model
history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)


# In[60]:



# Predict on the test set
y_pred = model.predict(test_X).flatten()


# In[61]:



# Calculate and print the RMSE, MAE, and R-squared score on the test set
print("RMSE on test set: ", mean_squared_error(test_y, y_pred, squared=False))
print("MAE on test set: ", mean_absolute_error(test_y, y_pred))
print("R-squared on test set: ", r2_score(test_y, y_pred))


# In[82]:


# Calculate and print the RMSE, MAE, and R-squared score on the test set
print("RMSE on test set: {:.2f}".format(mean_squared_error(test_y, y_pred, squared=False)))
print("MAE on test set: {:.2f}".format(mean_absolute_error(test_y, y_pred)))
print("R-squared on test set: {:.2f}".format(r2_score(test_y, y_pred)))


# In[ ]:





# In[84]:


import os
import pandas as pd

# Predict on test set
y_pred = model.predict(test_X).flatten()

# Create dataframe with actual and predicted values
df_results = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

# Create a filename for the Excel file
filename = 'actual_vs_predicted3.xlsx'

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




