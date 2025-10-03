"""
Problem 1: Feature Selection for Practice
Load Ames Housing dataset and select relevant features (GrLivArea, YearBuilt)
for regression practice. Saves a preview CSV in the data folder and updates README.
"""

import pandas as pd
import os

def main():
    print("Running Problem 1 - Feature Selection")

    # Paths
    data_folder = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_folder, exist_ok=True)
    dataset_path = os.path.join(data_folder, 'train.csv')
    preview_csv_path = os.path.join(data_folder, 'problem1_feature_selection_preview.csv')
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')

    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}. Please download train.csv from Kaggle.")
        return

    # Select features
    selected_features = ['GrLivArea', 'YearBuilt', 'SalePrice']
    df_selected = df[selected_features]

    # Optionally check missing values
    missing_values = df_selected.isnull().sum()
    print("Missing values per column:\n", missing_values)

    # Save preview CSV
    df_selected.head(10).to_csv(preview_csv_path, index=False)
    print(f"Preview of selected features saved to {preview_csv_path}")

    # Update README.md with table preview
    with open(readme_path, 'a') as f:
        f.write("\n## Problem 1 - Feature Selection\n")
        f.write("Selected features: GrLivArea, YearBuilt, SalePrice\n\n")
        f.write("Preview of first 10 rows:\n\n")
        f.write(df_selected.head(10).to_markdown(index=False))
        f.write("\n")

if __name__ == "__main__":
    main()
