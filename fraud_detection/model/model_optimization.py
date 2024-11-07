# model/model_optimization.py

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import optuna

# Function to optimize hyperparameters using GridSearchCV
def optimize_hyperparameters(X_train, y_train):
    # Example for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_

# Function for model training with ensemble methods
def train_ensemble_models(X_train, y_train):
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100)
    gb_model.fit(X_train, y_train)

    return rf_model, gb_model

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function to optimize models
def optimize_models(X_train, y_train, X_test, y_test):
    best_params = optimize_hyperparameters(X_train, y_train)
    print(f"Best Hyperparameters: {best_params}")

    rf_model, gb_model = train_ensemble_models(X_train, y_train)

    rf_accuracy = evaluate_model(rf_model, X_test, y_test)
    gb_accuracy = evaluate_model(gb_model, X_test, y_test)

    return {
        'Random Forest Accuracy': rf_accuracy,
        'Gradient Boosting Accuracy': gb_accuracy,
        'Best Hyperparameters': best_params
    }
