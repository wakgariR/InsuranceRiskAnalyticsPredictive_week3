import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple

def preprocess_data(df: pd.DataFrame, target_col: str, claims_only: bool = False) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """
    Preprocesses the data, handles claims-only filtering, encodes features, and returns split data.

    Args:
        df: The clean, full DataFrame.
        target_col: The prediction target ('TotalClaims' or 'HasClaim').
        claims_only: If True, filters data for Claim Severity Model (TotalClaims > 0).

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df_working = df.copy()

    # 1. Claims-Only Filtering for Severity Model
    if claims_only:
        df_working = df_working[df_working['HasClaim'] == 1].copy()
        
    # 2. Select Features (Excluding IDs, engineered targets, etc.)
    features_to_exclude = ['PolicyNumber', 'ClientNumber', 'ClaimNumber', 
                           'TotalPremium', 'Margin', 'HasClaim', 'CalculatedPremiumPerTerm']
    
    # Identify final feature set and target
    X = df_working.drop(columns=[col for col in features_to_exclude if col in df_working.columns] + [target_col], errors='ignore')
    y = df_working[target_col]

    # 3. Handle Missing Data (Final Imputation for modeling)
    # Using simple median/mode imputation for robustness
    X['CapitalOutstanding'] = X['CapitalOutstanding'].fillna(X['CapitalOutstanding'].median())
    X['SumInsured'] = X['SumInsured'].fillna(X['SumInsured'].median())
    X['NumberOfDoors'] = X['NumberOfDoors'].fillna(X['NumberOfDoors'].mode()[0])

    # 4. Feature Engineering (Target Encoding for Province/Zip Code)
    # We will use One-Hot Encoding for all categories to be explicit in this step.

    # 5. Define Feature Types for Preprocessing Pipeline
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    # Assuming Province, PostalCode, Gender, CoverType, etc., are the key categories
    categorical_features = ['Province', 'PostalCode', 'Gender', 'CoverType', 'Colour', 'VehicleType']
    categorical_features = [col for col in categorical_features if col in X.columns]
    
    # Remove categorical features that are numerical or not present
    numerical_features = [col for col in numerical_features if col not in categorical_features]


    # 6. Create Preprocessing Pipeline
    # Standardize numerical features and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Keep any other unlisted columns
    )
    
    # 7. Train-Test Split (80/20 standard)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if target_col == 'HasClaim' else None
    )

    return X_train, X_test, y_train, y_test, preprocessor

# Example Usage:
# from data_preparation import preprocess_data
# X_train_sev, X_test_sev, y_train_sev, y_test_sev, preprocessor_sev = preprocess_data(df_clean, 'TotalClaims', claims_only=True)
# X_train_prob, X_test_prob, y_train_prob, y_test_prob, preprocessor_prob = preprocess_data(df_clean, 'HasClaim', claims_only=False)