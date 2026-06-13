🚀 Real Estate Price Prediction Using Machine Learning | Mathematical Implementation of Multiple Linear Regression from Scratch

A Machine Learning project that predicts house prices based on property size, number of bedrooms, and house age.

Unlike most beginner projects that rely on pre-built machine learning models, this project implements **Multiple Linear Regression completely from scratch using NumPy**, including:

* Feature Scaling
* Cost Function Calculation
* Gradient Descent Optimization
* Parameter Updates
* Prediction System

No Scikit-Learn regression model was used during training.

---

## 🚀 Project Overview

This project demonstrates how machine learning algorithms work internally by manually implementing Multiple Linear Regression.

The model learns the relationship between:

* House Size
* Number of Bedrooms
* House Age

and predicts:

* House Price

using Gradient Descent optimization.

---

## 🧠 Mathematical Foundation

The prediction function is:

[
\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3
]

Where:

* (x_1) = Size
* (x_2) = Bedrooms
* (x_3) = Age

The model minimizes the Mean Squared Error Cost Function:

[
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}-y)^2
]

using Gradient Descent:

[
\theta_j = \theta_j - \alpha \frac{\partial J}{\partial \theta_j}
]

This project manually computes all gradients and updates parameters without using any machine learning library's regression implementation.

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn (only for train-test split and evaluation metrics)

---

## 📊 Model Evaluation

The model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

Example Output:

```text
MAE  : XXXXX
MSE  : XXXXX
RMSE : XXXXX
R²   : XXXXX
```

---

## 📈 Visualizations

### 1. Cost Reduction Over Time

Shows how Gradient Descent successfully minimizes the cost function during training.

### 2. Actual vs Predicted Prices

Compares the model's predictions with real house prices to evaluate performance.

---

## 🔮 Prediction System

After training, the user can enter:

* House Size
* Number of Bedrooms
* House Age

The model then predicts the estimated house price.

Example:

```text
Enter size: 1800
Enter bedrooms: 3
Enter age: 5

Predicted Price: 542,000
```

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── house_price_prediction.py
├── proce prediction model.csv
├── README.md
└── screenshots/
    ├── cost_reduction.png
    └── actual_vs_predicted.png
```

---

## 🏆 Key Learning Outcomes

* Understanding Linear Regression from first principles
* Implementing Gradient Descent manually
* Feature Scaling using Standardization
* Cost Function Optimization
* Model Evaluation Techniques
* Data Visualization for Machine Learning

---

## 💡 Future Improvements

* Vectorized implementation for faster training
* Regularization (L1/L2)
* Polynomial Regression
* Model Saving and Loading
* Web Deployment using Flask

---

### Built From Scratch With Mathematics, NumPy, and Gradient Descent — No Pre-Built Regression Models Used.
