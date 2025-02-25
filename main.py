"""
Hyperparameter Optimization using Optuna with RandomForestClassifier.

This script optimizes the hyperparameters of a RandomForestClassifier 
using Optuna's Tree-structured Parzen Estimator (TPE) on the Iris dataset.

Author: [Your Name]
"""

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# Load Iris dataset (for classification task)
data = load_iris()
X, y = data.data, data.target

def objective(trial):
    """
    Objective function for Optuna optimization.
    
    Parameters:
        trial (optuna.trial.Trial): A single optimization trial.

    Returns:
        float: The mean accuracy score of the model with the selected hyperparameters.
    """
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)

    # Define model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # Evaluate model using cross-validation
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return score

# Run optimization with 50 trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best results
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)
