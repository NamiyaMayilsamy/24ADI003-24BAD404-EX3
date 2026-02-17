print("Namiya-24BAD404")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

file_path = r"C:\Users\namiy\Downloads\auto-mpg.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower", "mpg"])

X = df[["horsepower"]].values
y = df["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

degrees = [1, 2, 3, 4, 10]
train_errors = []
test_errors = []
comparison_metrics = []

plt.figure(figsize=(20, 5))
for i, degree in enumerate([2, 3, 4]):
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train_scaled)
    X_poly_test = poly.transform(X_test_scaled)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)
    mse = mean_squared_error(y_test, y_pred)
    comparison_metrics.append({
        "Degree": degree,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, y_pred)
    })
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_test, y_test, alpha=0.5)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line_pred = model.predict(
        poly.transform(scaler.transform(X_line))
    )
    plt.plot(X_line, y_line_pred)
    plt.title(f"Degree {degree}")

plt.show()

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_p_train = poly.fit_transform(X_train_scaled)
    X_p_test = poly.transform(X_test_scaled)
    model = LinearRegression()
    model.fit(X_p_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_p_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_p_test)))

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, marker="o", label="Training Error")
plt.plot(degrees, test_errors, marker="o", label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
for i, d in enumerate([1, 10]):
    poly = PolynomialFeatures(degree=d)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train_scaled), y_train)
    plt.subplot(1, 2, i + 1)
    plt.scatter(X_test, y_test, alpha=0.5)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line_pred = model.predict(
        poly.transform(scaler.transform(X_line))
    )
    plt.plot(X_line, y_line_pred)
    plt.title(f"Degree {d}")

plt.show()

print(pd.DataFrame(comparison_metrics))

poly4 = PolynomialFeatures(degree=4)
ridge = Ridge(alpha=1.0)
ridge.fit(poly4.fit_transform(X_train_scaled), y_train)
y_ridge = ridge.predict(poly4.transform(X_test_scaled))
print("Ridge R2 Score:", r2_score(y_test, y_ridge))
