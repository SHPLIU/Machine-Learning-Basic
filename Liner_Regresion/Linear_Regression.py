import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dataframe = pd.read_fwf('kc_house_data4.txt')
x_values = dataframe[['sqft_living']]
y_values = dataframe[['price']]

# Split the data into training/testing sets
x_values_train = x_values[:-20]
x_values_test = x_values[-20:]

# Split the targets into training/testing sets
y_values_train = y_values[:-20]
y_values_test = y_values[-20:]

#visualize training and test data
plt.scatter(x_values_train, y_values_train, color='blue')
plt.scatter(x_values_test, y_values_test, color='green')
plt.show()

#train model on data
house_reg = linear_model.LinearRegression()
house_reg.fit(x_values_train, y_values_train)

# The coefficients
print('Coefficients: \n', house_reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((house_reg.predict(x_values_test) - y_values_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % house_reg.score(x_values_test, y_values_test))

#visualize regression results

plt.ylabel('price')
plt.xlabel('sqft_living')
plt.scatter(x_values_test, y_values_test, color='black')
plt.plot(x_values, house_reg.predict(x_values), color='red', linewidth=1)
plt.show()
