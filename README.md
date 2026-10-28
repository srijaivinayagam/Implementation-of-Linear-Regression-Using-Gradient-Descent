# Implementation-of-Linear-Regression-Using-Gradient-Descent

# AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Load the startup dataset.
2. Convert categorical data into numerical values.
3. Split the data into training and testing sets.
4. Initialize weights, bias, and learning rate.
5. Train the model using Gradient Descent.
6. Predict startup profits.
7. Calculate MSE and plot the cost function.
8. Display the results and stop.

# Program:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
```
## Step 1: Load the dataset
```
from google.colab import files
uploaded = files.upload()

df = pd.read_csv("exp_3_50_Startups.csv")

print("First 5 rows of the dataset:")
print(df.head())
```
## Step 2: Convert categorical data into numerical data
```
df = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=int)
```
## Step 3: Separate input and output variables
```
X = df.drop("Profit", axis=1).values
Y = df["Profit"].values.reshape(-1, 1)
```
## Step 4: Split the dataset
```
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
```
## Step 5: Scale the data
```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)
```
## Step 6: Initialize parameters
```
m, n = X_train.shape
weights = np.zeros((n, 1))
bias = 0.0
learning_rate = 0.01
epochs = 1000
cost_history = []
```
## Step 7: Perform gradient descent
```
for i in range(epochs):
    Y_pred = np.dot(X_train, weights) + bias

    error = Y_pred - Y_train
    dw = (2 / m) * np.dot(X_train.T, error)
    db = (2 / m) * np.sum(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    cost = np.mean(error ** 2)
    cost_history.append(cost)
```
## Step 8: Predict startup profits
```
Y_pred_scaled = np.dot(X_test, weights) + bias
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
```
## Step 9: Calculate Mean Squared Error
```
mse = mean_squared_error(Y_test, Y_pred)

print("\nActual Profits:\n", Y_test[:5].flatten())
print("\nPredicted Profits:\n", Y_pred[:5].flatten())
print("\nMean Squared Error:", round(mse, 2))
```
## Step 10: Plot the cost function
```
plt.figure(figsize=(8, 5))
plt.plot(cost_history)
plt.title("Gradient Descent Cost Function")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()

print("\nRESULT: Startup profit prediction using Gradient Descent was completed.")

```

# Output:
<img width="987" height="795" alt="image" src="https://github.com/user-attachments/assets/dc3a33cd-2719-476a-b1e4-a2a7d9ec44d1" />

# Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
