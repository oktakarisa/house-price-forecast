"""
Problem 2: Estimation and Evaluation by Linear Regression
Train a Linear Regression model on selected features, visualize predictions,
and update README.md with plot previews.
"""

import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")  # Prevent segmentation fault in GitBash
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    print("Running Problem 2 - Linear Regression")

    # Paths
    data_folder = os.path.join(os.path.dirname(__file__), '../data')
    plot_folder = os.path.join(os.path.dirname(__file__), '../plots')
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')
    os.makedirs(plot_folder, exist_ok=True)

    dataset_path = os.path.join(data_folder, 'train.csv')
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}. Please ensure train.csv exists in data/.")
        return

    # Select features and target
    X = df[['GrLivArea', 'YearBuilt']]
    y = df['SalePrice']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # Update README.md header
    with open(readme_path, 'a') as f:
        f.write("\n## Problem 2 - Linear Regression\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n\n")

    # Scatter plot and update README.md
    for feature in X.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(X_test[feature], y_test, color='blue', label='Actual')
        plt.scatter(X_test[feature], y_pred, color='red', alpha=0.6, label='Predicted')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(f"Linear Regression: {feature} vs SalePrice")
        plt.legend()
        plot_filename = f'problem2_linear_regression_{feature}.png'
        plot_path = os.path.join(plot_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")

        # Append markdown image link to README.md
        with open(readme_path, 'a') as f:
            f.write(f"**{feature} vs SalePrice:**  \n")
            f.write(f"![{feature}]({os.path.relpath(plot_path, os.path.dirname(readme_path)).replace('\\\\','/')})\n\n")

if __name__ == "__main__":
    main()
