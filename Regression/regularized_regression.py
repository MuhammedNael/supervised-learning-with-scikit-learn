'''
Regularized regression is a technique that is used to combat overfitting in regression models.
Regularized regression adds a penalty term to the loss function. This penalty term discourages the coefficients from being too large.
There are two types of regularized regression: Lasso and Ridge regression.
Performance gets worse when alpha increases. This is because the model is penalized for having large coefficients.
'''
# Import Ridge
from sklearn.linear_model import Ridge
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
  # Create a Ridge regression model
  ridge = Ridge(alpha=alpha)
  # Fit the data
  ridge.fit(X_train, y_train)
  # Obtain R-squared
  score = ridge.score(X_test, y_test)
  ridge_scores.append(score)
print(ridge_scores)

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)
# Fit the model to the data
lasso.fit(X, y)
# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()