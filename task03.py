
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("Housing.csv")
print("First 5 rows of data:")
print(df.head())
df = pd.get_dummies(df, drop_first=True)
print("\nMissing values:\n", df.isnull().sum())
X_simple = df[['area']]
y = df['price']
X_multi = df.drop('price', axis=1)
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)
y_pred_simple = model_simple.predict(X_test_simple)
y_pred_multi = model_multi.predict(X_test_multi)
print("\n Simple Linear Regression Metrics:")
print("MAE:", mean_absolute_error(y_test_simple, y_pred_simple))
print("MSE:", mean_squared_error(y_test_simple, y_pred_simple))
print("R² Score:", r2_score(y_test_simple, y_pred_simple))
print("\n Multiple Linear Regression Metrics:")
print("MAE:", mean_absolute_error(y_test_multi, y_pred_multi))
print("MSE:", mean_squared_error(y_test_multi, y_pred_multi))
print("R² Score:", r2_score(y_test_multi, y_pred_multi))
plt.figure(figsize=(8, 5))
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Predicted')
plt.title('Simple Linear Regression: Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n Simple Regression Coefficients:")
print("Intercept:", model_simple.intercept_)
print("Slope (area):", model_simple.coef_[0])

print("\n Multiple Regression Coefficients:")
print("Intercept:", model_multi.intercept_)
print("Coefficients:", pd.Series(model_multi.coef_, index=X_multi.columns))