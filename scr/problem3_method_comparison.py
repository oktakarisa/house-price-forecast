"""
Problem 3: Comparison of Regression Methods
Train Linear Regression, SVR, Decision Tree, and Random Forest on selected features,
evaluate MSE, visualize results, and update README.md with table and plots.
"""

import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    print("Running Problem 3 - Method Comparison")

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

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    # Store results
    results = []

    # Train, predict, evaluate, plot
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE for {name}: {mse:.2f}")
        results.append((name, mse))

        # Scatter plot for each feature
        for feature in X.columns:
            plt.figure(figsize=(6,4))
            plt.scatter(X_test[feature], y_test, color='blue', label='Actual')
            plt.scatter(X_test[feature], y_pred, color='red', alpha=0.6, label='Predicted')
            plt.xlabel(feature)
            plt.ylabel('SalePrice')
            plt.title(f"{name}: {feature} vs SalePrice")
            plt.legend()
            plot_filename = f'problem3_{name.lower().replace(" ", "_")}_{feature}.png'
            plot_path = os.path.join(plot_folder, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved: {plot_path}")

            # Update README.md with markdown preview
            with open(readme_path, 'a') as f:
                f.write(f"**{name} - {feature} vs SalePrice:**  \n")
                f.write(f"![{feature}]({os.path.relpath(plot_path, os.path.dirname(readme_path)).replace('\\\\','/')})\n\n")

    # Update README.md with results table
    with open(readme_path, 'a') as f:
        f.write("\n### Problem 3 – Method Comparison\n")
        f.write("| Method | MSE |\n")
        f.write("|--------|------|\n")
        for name, mse in results:
            f.write(f"| {name} | {mse:.2f} |\n")
        f.write("\n")

if __name__ == "__main__":
    main()
