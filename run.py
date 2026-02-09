import pandas as pd                      # for data handling
import numpy as np                       # for numerical operations
import matplotlib.pyplot as plt          # for plotting
from sklearn.linear_model import LinearRegression  # linear regression model

# Load the dataset
df = pd.read_csv("Salary.csv")           # load Salary dataset

# Separate independent and dependent variables
X = df[['years_of_experience']]          # independent variable
y = df['salary']                         # dependent variable

# Create linear regression model
model = LinearRegression()               # initialize model

# Train model using complete dataset
model.fit(X, y)                          # fit model on full data

# Predict salary using trained model
y_pred = model.predict(X)                # predicted salaries

# Plot best fit line with data points
plt.figure()
plt.scatter(X, y)                        # plot actual data points
plt.plot(X, y_pred)                      # plot best fit line
plt.xlabel("Years of Experience")        # x-axis label
plt.ylabel("Salary")                     # y-axis label
plt.title("Simple Linear Regression: Salary vs Experience")
plt.show()                               # display plot

# Residual plot
residuals = y - y_pred                   # calculate residuals

plt.figure()
plt.scatter(X, residuals)                # plot residuals
plt.axhline(y=0)                         # reference line at zero
plt.xlabel("Years of Experience")        # x-axis label
plt.ylabel("Residuals")                  # y-axis label
plt.title("Residual Plot")
plt.show()   