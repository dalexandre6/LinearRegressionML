# Simple Linear Regression Between Income and Money expent in tech products:

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


# Importing the dataset as csv:
dataset = pd.read_csv('LG_Income.csv')
#X must be a matix (not a vector):
X = dataset.iloc[:, :-2].values
#y can be a vectos:
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set.
#For this, we use sklearn with model_selection, then the Splitter Function:
#This functions splits arrays or matrices. Try to split (80% to 20%):
#80% for training set and 20% for test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Check vectors and Matrices:
print(X.shape), print(X_train.shape), print(X_test.shape),
print(y_train.shape), print(y_test.shape), print(y.shape)


# Fitting the Linear Regression to the Training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Make your predictions on the Test Set:

regressor.predict(65000)
regressor.predict(15000)
regressor.predict(160000)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Net Income vs Tech Expenses (Training set)')
plt.xlabel('Net Income')
plt.ylabel('$ Tech Expenses ')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Net Income vs Tech Expenses (Test set)')
plt.xlabel('Net Income')
plt.ylabel('$ Tech Expenses')
plt.show()


#Statistic Analysis:
import statsmodels.api as sm
#Create object:
#Tell python about your variables: dependent(y) and then the independent(X) in 
#this case. You may have others.
#OLS means Ordinary Least Square(Just the type of linear regression):
model = sm.OLS(y,X) 
result = model.fit()


#Define y = a + bx to prove the model: 
# slope  (b)
print(regressor.coef_) 
# intercept(a)
print(regressor.intercept_) 
#Then replace the variable x with the values you wish to predict
#For instance:  a + b(65000)  --> this will give you the y value
#for 65000 dollars in net income.