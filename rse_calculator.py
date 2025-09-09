# Function definition with three parameters:
# y_actual: array of actual/observed values
# y_predicted: array of predicted values from the model
# p: number of predictors/features used in the model
def rse_calculator(y_actual, y_predicted, p):

  # Calculate Residual Standard Error (RSE):
  # 1. (y_actual - y_predicted): Calculate residuals (errors)
  # 2. **2: Square each residual to eliminate negative signs
  # 3. np.sum(): Sum all squared residuals (Sum of Squared Errors - SSE)
  # 4. / (y_actual.size - p - 1): Divide by degrees of freedom
  #    - y_actual.size: total number of observations (n)
  #    - p: number of predictors
  #    - 1: accounts for the intercept term
  #    - Degrees of freedom = n - p - 1
  # 5. np.sqrt(): Take square root to return to original units
  rse_value = np.sqrt(np.sum((y_actual - y_predicted)**2) / (y_actual.size - p - 1))

  # Return RSE value rounded to 2 decimal places for readability
  return np.round(rse_value, 2)