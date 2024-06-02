# Importing necessary libraries
import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load data
dir_path = os.getcwd()
mat = scipy.io.loadmat(fr"../Matlab/people10000_person.mat")
rssi_values_matrix = np.array(np.abs(mat['RSSI']), dtype=np.double)
people_position = np.array(np.abs(mat['positions']), dtype=np.double)[:, :2]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(rssi_values_matrix, people_position, test_size=0.2, random_state=42)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=4)  # You can adjust the degree of the polynomial
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

# Training the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# Training the Ridge regression model
model = Ridge(alpha=1.0)  # You can adjust the alpha parameter for regularization
model.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Printing the coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
