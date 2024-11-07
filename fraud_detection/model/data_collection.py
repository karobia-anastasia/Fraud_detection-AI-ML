import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import optuna

def load_cleaned_data(file_path):
    return pd.read_csv(file_path)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 1, 32)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    X = data.drop('isFraud', axis=1)
    y = data['isFraud']
    
    score = cross_val_score(clf, X, y, n_jobs=-1, cv=3)
    return score.mean()

if __name__ == "__main__":
    file_path = 'cleaned_creditcard.csv'
    data = load_cleaned_data(file_path)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
    X = data.drop('isFraud', axis=1)
    y = data['isFraud']
    final_model.fit(X, y)

    joblib.dump(final_model, 'optimized_model.pkl')
