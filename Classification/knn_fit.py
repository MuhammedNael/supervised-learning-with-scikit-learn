''' 
In this exercise, you will build your first classification model using the churn_df dataset, which has been preloaded 
for the remainder of the chapter.
The target, "churn", needs to be a single column with the same number of observations as the feature data. 
The feature data has already been converted into numpy arrays.
'''

from numpy import np
# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(6)

# Fit the classifier to the data
knn.fit(X, y)


X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])


# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred)) 