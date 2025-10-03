"""
Problem 4: Advanced - Learning Using Additional Features
Select more features, preprocess data, train multiple models, evaluate MSE,
generate plots, and update README.md with results.
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
from sklearn.preprocessing import StandardScaler

def main():
    print("Running Problem 4 - Additional Features")

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

    # Select additional features
    additional_features = ['GrLivArea', 'YearBuilt', 'OverallQual', 'TotRmsAbvGrd', 'LotArea']
    # Handle missing values: simple fill with median for numeric
    df_selected = df[additional_features + ['SalePrice']].copy()
    df_selected.fillna(df_selected.median(numeric_only=True), inplace=True)

    # Separate X and y
    X = df_selected[additional_features]
    y = df_selected['SalePrice']

    # Encode categorical variables if any (none in these features, but placeholder)
    # X = pd.get_dummies(X)

    # Scale features for SVR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

        # Scatter plot for each feature (scaled)
        for i, feature in enumerate(additional_features):
            plt.figure(figsize=(6,4))
            plt.scatter(X_test[:, i], y_test, color='blue', label='Actual')
            plt.scatter(X_test[:, i], y_pred, color='red', alpha=0.6, label='Predicted')
            plt.xlabel(feature)
            plt.ylabel('SalePrice')
            plt.title(f"{name}: {feature} vs SalePrice")
            plt.legend()
            plot_filename = f'problem4_{name.lower().replace(" ", "_")}_{feature}.png'
            plot_path = os.path.join(plot_folder, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved: {plot_path}")

            # Update README.md with markdown image preview
            with open(readme_path, 'a') as f:
                f.write(f"**{name} - {feature} vs SalePrice:**  \n")
                f.write(f"![{feature}]({os.path.relpath(plot_path, os.path.dirname(readme_path)).replace('\\\\','/')})\n\n")

    # Update README.md with results table
    with open(readme_path, 'a') as f:
        f.write("\n### Problem 4 - Additional Features\n")
        f.write("| Method | MSE |\n")
        f.write("|--------|------|\n")
        for name, mse in results:
            f.write(f"| {name} | {mse:.2f} |\n")
        f.write("\n")
        f.write("Comment: Adding more features generally improves model performance, "
                "and tree-based models can help identify feature importance.\n")

if __name__ == "__main__":
    main()
