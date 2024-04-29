'''
R^2 may not be the best metric to evaluate the model's performance. Because it is not representative of the model's performance on unseen data.
To address this, you can use cross-validation. Cross-validation is a technique used to evaluate the model's performance on unseen data.
'''

# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)
# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print((np.std(cv_scores)))

# Print the 95% confidence interval
print(np.quantile(cv_scores, [0.025, 0.975]))