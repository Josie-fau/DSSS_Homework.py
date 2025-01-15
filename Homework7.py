import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

## Classification
file_path = "C:/Users/jojor/Desktop/Uni/Master/2425WS/DSSS/HW7/classification.csv"
classification = pd.read_csv(file_path)

classification_1 = classification[classification['label'] == 1]
classification_0 = classification[classification['label'] == 0]

# Create the scatter plot
plt.scatter(classification_1['x1'], classification_1['x2'], color='blue', label='Label 1')
plt.scatter(classification_0['x1'], classification_0['x2'], color='purple', label='Label 0')

# Add plot labels and legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Classification')

# Show the plot
plt.show()

## Regression 1
file_path = "C:/Users/jojor/Desktop/Uni/Master/2425WS/DSSS/HW7/regression_1.csv"
regression1 = pd.read_csv(file_path)

# Create the scatter plot
plt.scatter(regression1['x1'], regression1['x2'], color='green')

# Add plot labels and legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Regression 1')

# Show the plot
plt.show()

## Regression 2
file_path = "C:/Users/jojor/Desktop/Uni/Master/2425WS/DSSS/HW7/regression_2.csv"
regression2 = pd.read_csv(file_path)

# Create the scatter plot
plt.scatter(regression2['x1'], regression2['x2'], color='green')

# Add plot labels and legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Regression 2')

# Show the plot
plt.show()


## ML classification
# features (X) and target (y)
X_class = classification[['x1', 'x2']].values
y_class = classification['label'].values
# Split training and test sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_class, y_train_class)
xx, yy = np.meshgrid(np.linspace(X_class[:, 0].min(), X_class[:, 0].max(), 100),
                     np.linspace(X_class[:, 1].min(), X_class[:, 1].max(), 100))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train_class[:, 0], X_train_class[:, 1], c=y_train_class, cmap='coolwarm', marker='o', edgecolor='k', s=50, label='Original Data (Train)')
plt.scatter(X_test_class[:, 0], X_test_class[:, 1], c=y_test_class, cmap='coolwarm', marker='x', edgecolor='k', s=100, label='Original Data (Test)')
plt.title("Logistic Regression - Decision Boundary (Classification)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc='best')
plt.colorbar(label="Label")
plt.show()


## Regression1
# features (X) and target (y)
X_reg = regression1[['x1']].values  # Feature (x1)
y_reg = regression1['x2'].values    # Target (x2)
# Split training and test sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
# Polynomial Feature Transformation (degree 3 for cubic)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train_reg)  # Transform the training data
X_poly_test = poly.transform(X_test_reg)        # Transform the test data
# Train
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train_reg)
# Predict
y_pred_train = poly_reg.predict(X_poly_train)
y_pred_test = poly_reg.predict(X_poly_test)
# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train_reg, y_train_reg, color='blue', label='Original Data (Train)')
plt.scatter(X_test_reg, y_test_reg, color='red', label='Original Data (Test)')
# Generate fit line
x_range = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
x_poly_range = poly.transform(x_range)  # Apply polynomial transformation to the range
y_poly_range = poly_reg.predict(x_poly_range)
# Plot fit line
plt.plot(x_range, y_poly_range, color='green', label='Results obtained with Polynomial Regression')
plt.title("Polynomial Regression (Degree 3) - Fit Line")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

##Regression 2
# features (X) and target (y)
X_reg2 = regression2[['x1']].values  # Feature (x1)
y_reg2 = regression2['x2'].values    # Target (x2)
# Split training and test sets
X_train_reg2, X_test_reg2, y_train_reg2, y_test_reg2 = train_test_split(X_reg2, y_reg2, test_size=0.3, random_state=42)
# Polynomial Feature Transformation (degree 3 for cubic)
poly2 = PolynomialFeatures(degree=3)  # Use degree 3 for cubic fit
X_poly_train2 = poly2.fit_transform(X_train_reg2)  # Transform the training data
X_poly_test2 = poly2.transform(X_test_reg2)        # Transform the test data
# Train
poly_reg2 = LinearRegression()
poly_reg2.fit(X_poly_train2, y_train_reg2)
# Predict
y_pred_train2 = poly_reg2.predict(X_poly_train2)
y_pred_test2 = poly_reg2.predict(X_poly_test2)
# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train_reg2, y_train_reg2, color='blue', label='Original Data (Train)')
plt.scatter(X_test_reg2, y_test_reg2, color='red', label='Original Data (Test)')
#fitted polynomial line
x_range2 = np.linspace(X_reg2.min(), X_reg2.max(), 100).reshape(-1, 1)  # Range of x1 values
x_poly_range2 = poly2.transform(x_range2)  # Apply polynomial transformation
y_poly_range2 = poly_reg2.predict(x_poly_range2)  # Get the predicted y values
plt.plot(x_range2, y_poly_range2, color='green', label='Results obtained with Polynomial Regression')
plt.title("Polynomial Regression (Degree 3) - Fitted Line (Regression2)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()


