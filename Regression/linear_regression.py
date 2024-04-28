'''
In this chapter, you will work with a dataset called sales_df, which contains information on advertising campaign
expenditure across different media types, and the number of dollars generated in sales for the respective campaign. 
The dataset has been preloaded for you. Here are the first two rows:

     tv        radio      social_media    sales
1    13000.0   9237.76    2409.57         46677.90
2    41000.0   15886.45   2913.41         150177.83

You will use the advertising expenditure as features to predict sales values, initially working with the "radio" column. 
However, before you make any predictions you will need to create the feature and target arrays, reshaping them to the 
correct format for scikit-learn.
'''

import numpy as np

# Create X from the radio column's values
X = sales_df["radio"].values
# Create y from the sales column's values
y = sales_df["sales"].values
# Reshape X
X = X.reshape(-1, 1)
# Check the shape of the features and targets
print(y.shape, X.shape)


# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Create the model
reg = LinearRegression()
# Fit the model to the data
reg.fit(X, y)
# Make predictions
predictions = reg.predict(X)
print(predictions[:5])


# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()
