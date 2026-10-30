# SGD Regressor for Multivariate Linear Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the California housing dataset.
2. Take the features as X (all columns except AveOccup and the house price) and the targets as Y (AveOccup and the house price).
3. Split the data into training and testing sets and standardize the features and targets using StandardScaler.
4. Create an SGDRegressor, wrap it in MultiOutputRegressor, and train it on the training data.
5. Predict the values for the test data, inverse-transform them, and calculate the Mean Squared Error.

## Program:
```python
'''
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: <S.DURGESH>
RegisterNumber: <212225230064>
'''
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["HousingPrice"] = data.target
print(df.head())

X = df.drop(columns=["AveOccup", "HousingPrice"])
Y = df[["AveOccup", "HousingPrice"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```

## Output:
<img width="782" height="372" alt="image" src="https://github.com/user-attachments/assets/9d0b7747-47ce-4e21-b0ca-21a2397b108c" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
