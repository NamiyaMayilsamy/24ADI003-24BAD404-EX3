print("Namiya-24BAD404")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\namiy\Downloads\StudentsPerformance.csv")

df["final_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

np.random.seed(0)
df["study_hours"] = np.random.randint(1, 10, len(df))
df["attendance"] = np.random.randint(60, 100, len(df))
df["sleep_hours"] = np.random.randint(4, 9, len(df))

df = pd.get_dummies(df, drop_first=True)

X = df.drop(["math score", "reading score", "writing score", "final_score"], axis=1)
y = df["final_score"]

X = X.apply(pd.to_numeric, errors="coerce")
X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nRegression Coefficients:")
print(coefficients)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("\nRidge R2 Score:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

lasso_coeff = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lasso.coef_
})

print("\nLasso Coefficients:")
print(lasso_coeff)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Predicted vs Actual Exam Scores")
plt.show()

plt.figure(figsize=(10,6))
plt.barh(coefficients["Feature"], coefficients["Coefficient"])
plt.title("Coefficient Magnitude Comparison")
plt.show()

residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
