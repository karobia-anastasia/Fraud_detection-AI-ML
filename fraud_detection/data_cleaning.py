import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(data):
    """Preprocess the data by handling missing values, normalizing, and feature engineering."""
    
    print("Missing values before preprocessing:\n", data.isnull().sum())
    
    # Drop duplicates
    data = data.drop_duplicates()

    # Identify numeric and categorical features
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    categorical_features = ['type']

    # Check that the expected columns exist
    for col in numeric_features + categorical_features:
        if col not in data.columns:
            raise ValueError(f"The column '{col}' is missing from the data.")

    # Print data types to debug issues
    print("Data types before preprocessing:\n", data.dtypes)

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
                ('scaler', StandardScaler())  # Normalize numeric features
            ]), numeric_features),
            
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing categorical values
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # One-hot encode categorical features
            ]), categorical_features)
        ],
        remainder='drop'  # Drop any columns that are not listed in the transformers
    )

    # Apply preprocessing
    processed_data = preprocessor.fit_transform(data)

    # Get feature names after transformation
    numeric_feature_names = numeric_features
    categorical_feature_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))

    # Combine feature names
    feature_names = numeric_feature_names + categorical_feature_names

    # Convert back to DataFrame
    processed_data = pd.DataFrame(processed_data, columns=feature_names)

    print("Data shape after preprocessing:", processed_data.shape)
    print("\nMissing values after preprocessing:\n", processed_data.isnull().sum())

    return processed_data
