"""
Main runner for house-price-forecast project.
Executes all problem scripts in order.
"""

import sys
import os

# Ensure 'scr' folder is in the Python path
scr_path = os.path.join(os.path.dirname(__file__), 'scr')
if scr_path not in sys.path:
    sys.path.append(scr_path)

# Import problem scripts
try:
    import problem1_feature_selection
    import problem2_linear_regression
    import problem3_method_comparison
    import problem4_additional_features
except ModuleNotFoundError as e:
    print(f"Error importing scripts: {e}")
    sys.exit(1)

def main():
    print("\n=== Running House Price Forecast Assignment Scripts ===\n")
    
    print(">> Problem 1: Feature Selection")
    problem1_feature_selection.main()
    
    print("\n>> Problem 2: Linear Regression")
    problem2_linear_regression.main()
    
    print("\n>> Problem 3: Method Comparison")
    problem3_method_comparison.main()
    
    print("\n>> Problem 4: Additional Features")
    problem4_additional_features.main()
    
    print("\n=== All Scripts Executed Successfully ===\n")

if __name__ == "__main__":
    main()
