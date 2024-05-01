# Fine-Tuning is a process of adjusting the hyperparameters of a model to improve its performance.
# Hyperparameters are parameters that are set before the learning process begins.

# confusion matrix is a table that is often used to describe the performance of a classification model 
# on a set of data for which the true values are known.

# use of diabetes dataset
# Import confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))