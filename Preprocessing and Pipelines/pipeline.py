'''
Pipelines are a way to streamline the machine learning process. They are a way to chain multiple estimators into one. 
This is useful as there is often a fixed sequence of steps in processing the data, for example feature selection, 
normalization and classification. Pipelines help to prevent data leakage in your test harness by ensuring that 
data preparation like standardization is constrained to each fold of your cross validation procedure. 
The example below demonstrates the pipeline defined with two steps:'''

# Import modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Instantiate an imputer
imputer = SimpleImputer()

# Instantiate a knn model
knn = KNeighborsClassifier(3)

# Build steps for the pipeline
steps = [("imputer", imputer), 
         ("knn", knn)]

steps = [("imputer", imp_mean),
        ("knn", knn)]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))