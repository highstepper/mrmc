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

