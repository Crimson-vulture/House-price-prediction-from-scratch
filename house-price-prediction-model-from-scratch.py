import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv(r"C:\Users\ADITYA\Desktop\Jupyter\ML-projects\price prediction model.csv")

# Features and target
X = data[["size", "bedrooms", "age"]].values
y = data["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature scaling (important)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Initialize parameters
t0 = 0
t1 = 0
t2 = 0
t3 = 0

learning_rate = 0.01
iterations = 5000

m = len(y_train)

cost_history = []

# Training loop
for i in range(iterations):

    y_pred = (
        t0
        + t1 * X_train[:, 0]
        + t2 * X_train[:, 1]
        + t3 * X_train[:, 2]
    )

    error = y_pred - y_train

    # Gradients
    d_t0 = (1 / m) * np.sum(error)
    d_t1 = (1 / m) * np.sum(error * X_train[:, 0])
    d_t2 = (1 / m) * np.sum(error * X_train[:, 1])
    d_t3 = (1 / m) * np.sum(error * X_train[:, 2])

    # Update
    t0 -= learning_rate * d_t0
    t1 -= learning_rate * d_t1
    t2 -= learning_rate * d_t2
    t3 -= learning_rate * d_t3

    # Cost function
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    cost_history.append(cost)

# Predictions on test data
y_test_pred = (
    t0
    + t1 * X_test[:, 0]
    + t2 * X_test[:, 1]
    + t3 * X_test[:, 2]
)

# Metrics
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print("\nModel Evaluation on Test Data")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# Print learned parameters
print("\nParameters:")
print("t0:", t0)
print("t1:", t1)
print("t2:", t2)
print("t3:", t3)

# Plot cost vs iterations
plt.figure(figsize=(8,5))
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Time")
plt.grid(True)
plt.show()

# Actual vs Predicted Graph
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

# Prediction
new_size = float(input("Enter size: "))
new_bedrooms = float(input("Enter bedrooms: "))
new_age = float(input("Enter age: "))

# Scale input using SAME mean & std
new_X = np.array([new_size, new_bedrooms, new_age])
new_X = (new_X - mean) / std

prediction = (
    t0
    + t1 * new_X[0]
    + t2 * new_X[1]
    + t3 * new_X[2]
)

print(f"\nPredicted Price: {prediction:,.2f}")
