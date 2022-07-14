import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.txt", sep="|")
df.head()
# By inspection of data.txt, the only NAs occur for inflation and interest in 1950 Q1
# data.txt includes complete quarterly data from 1950 Q1 to 2000 Q4 
# with all dates in order and with no missing quarters

# Only consumption, dpi and unemp are required for this analysis
# The rest can be dropped, including date labels
# Regression is to be performed on differences of the macroeconomic data values
dconsum = (df.consumption - df.consumption.shift())[1:]
ddpi = (df.dpi - df.dpi.shift())[1:]
dunemp = (df.unemp - df.unemp.shift())[1:]

# visualize consumption vs dpi
fig, ax = plt.subplots()
ax.scatter(ddpi, dconsum)
plt.show()

# visualize consumption vs unemployment
fig, ax = plt.subplots()
ax.scatter(dunemp, dconsum)
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
df = pd.read_csv("data.txt", sep="|")
df.head()
# By inspection of data.txt, the only NAs occur for inflation and interest in 1950 Q1
# data.txt includes complete quarterly data from 1950 Q1 to 2000 Q4 
# with all dates in order and with no missing quarters

# Only consumption, dpi and unemp are required for this analysis
# The rest can be dropped, including date labels
# Regression is to be performed on differences of the macroeconomic data values
dconsum = (df.consumption - df.consumption.shift())[1:]
ddpi = (df.dpi - df.dpi.shift())[1:]
dunemp = (df.unemp - df.unemp.shift())[1:]

# Visualize consumption vs dpi
fig, ax = plt.subplots()
ax.scatter(ddpi, dconsum)
plt.show()

# Visualize consumption vs unemployment
fig, ax = plt.subplots()
ax.scatter(dunemp, dconsum)
plt.show()

# Visualize independence of dpi and unemployment
fig, ax = plt.subplots()
ax.scatter(dunemp, ddpi)
plt.show()

# Task 1: Replication of Results 
# Set up regression inputs for statmodels.api
y = pd.DataFrame(dconsum)
X = pd.DataFrame(ddpi)
X['unemp'] = dunemp
X = sm.add_constant(X) ## Add an intercept (beta_0) to the model


model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()

# Note that regression coefficients, standard errors, t-statistics, and p-values 
# match given values from the coding exercise.
# The R-squared result of 33.5% does not indicate a good fit.

# Now fit using scikit-learn
reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.coef_

reg.intercept_

# Task 2: Outlier Detections
# Examine the time series of residuals
predictions = reg.predict(X)
residuals = (y - predictions)

# Plot the residuals time series
fig, ax = plt.subplots()
ax.plot(residuals)
plt.show()

# First quartile (Q1)
Q1 = np.percentile(residuals, 25, interpolation = 'midpoint')

# Median (Q2)
Q2 = np.percentile(residuals, 50, interpolation = 'midpoint')
  
# Third quartile (Q3)
Q3 = np.percentile(residuals, 75, interpolation = 'midpoint')
  
# Interquaritle range (IQR)
IQR = Q3 - Q1

# Tukey lower and upper bounds

residuals




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
df = pd.read_csv("data.txt", sep="|")
df.head()
# By inspection of data.txt, the only NAs occur for inflation and interest in 1950 Q1
# data.txt includes complete quarterly data from 1950 Q1 to 2000 Q4 
# with all dates in order and with no missing quarters

# Only consumption, dpi and unemp are required for this analysis
# The rest can be dropped, including date labels
# Regression is to be performed on differences of the macroeconomic data values
dconsum = (df.consumption - df.consumption.shift())[1:]
ddpi = (df.dpi - df.dpi.shift())[1:]
dunemp = (df.unemp - df.unemp.shift())[1:]

# Visualize consumption vs dpi
fig, ax = plt.subplots()
ax.scatter(ddpi, dconsum)
plt.show()

# Visualize consumption vs unemployment
fig, ax = plt.subplots()
ax.scatter(dunemp, dconsum)
plt.show()

# Visualize independence of dpi and unemployment
fig, ax = plt.subplots()
ax.scatter(dunemp, ddpi)
plt.show()

# Task 1: Replication of Results 
# Set up regression inputs for statmodels.api
y = pd.DataFrame(dconsum)
X = pd.DataFrame(ddpi)
X['unemp'] = dunemp
X = sm.add_constant(X) ## Add an intercept (beta_0) to the model


model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()

# Note that regression coefficients, standard errors, t-statistics, and p-values 
# match given values from the coding exercise.
# The R-squared result of 33.5% does not indicate a good fit.

# Now fit using scikit-learn
reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.coef_

reg.intercept_

# Task 2: Outlier Detections
# Examine the time series of residuals
predictions = reg.predict(X)
residuals = (y - predictions)

# Plot the residuals time series
fig, ax = plt.subplots()
ax.plot(residuals)
plt.show()

# First quartile (Q1)
Q1 = np.percentile(residuals, 25, interpolation = 'midpoint')
Q1

# Median (Q2)
Q2 = np.percentile(residuals, 50, interpolation = 'midpoint')
Q2

# Third quartile (Q3)
Q3 = np.percentile(residuals, 75, interpolation = 'midpoint')
Q3

# Interquaritle range (IQR)
IQR = Q3 - Q1
IQR

# Tukey lower and upper bounds
Tlower = Q1 - 1.5*IQR
Tlower

Tupper = Q3 + 1.5*IQR
Tupper

r = np.array(residuals)
l = np.count_nonzero(r < Tlower)
l

u = np.count_nonzero(r > Tupper)
u


n = len(r)
n

# Conclusions for Task 2
# There are 12 outliers in the dataset with 203 elements, representing 6% of the data
# There are 3x outliers on the upper side vs lower: 9 upper vs 3 lower.
# Observing the plot in I11, the upper outliers happen in the later half of the time series.
# This imbalance implies a bias which invalidates the regression results.
# The cause of this problem is that consumption and dpi figures increase exponentially 
# over the 50 year period, and so differences between quarters grow over time.
# It is more appropriate to do the analysis with log returns or percent returns
# in order to remove this bias.

# Test 3: Autocorrelation of Residuals
# Calculate the Durbin-Watson (DW) statistic of the residuals

# Quarterly differences of residuals
dresiduals = (residuals - residuals.shift())[1:]
numer = (dresiduals**2).sum()

# Sum of the quared residuals
denom = (residuals**2).sum()

# DW statistic
DW = numer / denom
DW

# Conclusions for Test 3
# The impact of autocorrelation of the errors on the regression is

# Task 4: Bootstapping of Standard Errors
# Recalculate the standard errors of the regression coefficients using bootstrapping.


