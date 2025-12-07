import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visualization styles
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 6)

def get_column_types(df: pd.DataFrame, categorical_threshold: int = 50) -> tuple:
    """
    Identifies numerical and categorical columns. PostalCode and RegistrationYear 
    are treated as categorical if their unique count is below a threshold.
    """
    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        # Exclude IDs from being plotted
        if 'ID' in col or 'Code' in col or 'mmcode' in col or 'Year' in col:
            # Treat high-cardinality codes as categorical for grouping later, 
            # but don't try to plot bar charts for PostalCode (too many unique values)
            if df[col].nunique() < categorical_threshold:
                 categorical_cols.append(col)
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for binary features (which should be treated as categorical)
            if df[col].nunique() <= 2:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            # Use a threshold to avoid plotting bar charts for columns like PostalCode or Model with millions of unique values
            if df[col].nunique() < categorical_threshold:
                categorical_cols.append(col)
            else:
                # Flag high-cardinality object columns (like Model or Make) 
                # to be handled separately or ignored in a full bar chart plot
                pass
                
    return numerical_cols, categorical_cols

# Assuming df_clean is your final processed DataFrame
# numerical_features, categorical_features = get_column_types(df_clean)

def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: list):
    """
    Plots histograms and Box plots for numerical variables.
    """
    print("\n## 📈 Histograms (Numerical Variables)")
    
    # Define a subset of key columns for focused analysis
    key_numerical = ['TotalClaims', 'TotalPremium', 'SumInsured', 'ExcessSelected', 'VehicleAge']
    cols_to_plot = [col for col in key_numerical if col in numerical_cols]

    for col in cols_to_plot:
        # Create a figure with two subplots: histogram and box plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram (Density Plot)
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {col}', fontsize=14)
        axes[0].set_xlabel(col)
        axes[0].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        axes[0].legend()

        # Box Plot (for identifying outliers) 

        #[Image of Box Plot]

        sns.boxplot(x=df[col].dropna(), ax=axes[1])
        axes[1].set_title(f'Box Plot of {col} (Outliers)', fontsize=14)

        plt.tight_layout()
        plt.show()

# Example usage:
# plot_numerical_distributions(df_clean, numerical_features)

def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list):
    """
    Plots bar charts (Count Plots) for categorical variables.
    Focuses on variables with fewer unique categories for clear visualization.
    """
    print("\n## 📊 Bar Charts (Categorical Variables)")
    
    # Define a subset of key categorical columns for hypothesis testing
    key_categorical = ['Province', 'Gender', 'MaritalStatus', 'AlarmImmobiliser', 'TrackingDevice', 'LegalType']
    cols_to_plot = [col for col in key_categorical if col in categorical_cols and df[col].nunique() < 20] # Limit to 20 unique values for readability

    for col in cols_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Calculate value counts
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        
        # Plot count plot
        sns.barplot(x='Count', y=col, data=value_counts, palette="viridis")
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

# Example usage:
# plot_categorical_distributions(df_clean, categorical_features)