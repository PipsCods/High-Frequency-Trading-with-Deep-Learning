import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path


# Functions

def create_lag_features(df: pd.DataFrame, lags: int, target_col: str = 'return') -> pd.DataFrame:
    """Add lag features per symbol and date."""
    df = df.copy()
    df = df.sort_values(['SYMBOL', 'DATE', 'TIME'])
    for lag in range(1, lags + 1):
        df[f'{target_col}_lag_{lag}'] = df.groupby(['SYMBOL', 'DATE'])[target_col].shift(lag)
    df = df.dropna(subset=[f'{target_col}_lag_{lag}' for lag in range(1, lags + 1)])
    return df

def data_split(df: pd.DataFrame,
               date_col: str,
               target: str,
               features: list = None,
               pt: float = 0.8,
               standardize : bool = False) -> tuple:
    df = df.dropna()
    df = df.sort_values(date_col).set_index(date_col)
    
    X = df[features]
    Y = df[target]

    #split the data sequentially since it is time series data
    X_train = X.iloc[:int(pt*len(X)), :]
    X_test = X.iloc[int(pt*len(X)):, :]
    Y_train = Y.iloc[:int(pt*len(Y))]
    Y_test = Y.iloc[int(pt*len(Y)):]

    if standardize: #Standardize the features usign the training data
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train - mean)/std
        X_test = (X_test - mean)/std

    #check if the shapes are correct
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]

    return X_train, X_test, Y_train, Y_test

def regression(df: pd.DataFrame,
               date_col: str,
               target: str,
               features: list,
               alpha: float = 0.1,
               model: str = 'ridge',
               standardize : bool = False,
               pt : float = 0.8 ) -> dict:
    
    X_train, X_test, y_train, y_test = data_split(df = df,date_col=date_col, target=target, features=features,pt= pt, standardize=standardize)

    if model == 'ridge':
        model = Ridge(alpha=alpha)
    elif model == 'lasso':
        model = Lasso(alpha=alpha)
    elif model == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("Model must be 'ridge' or 'lasso'")
    
    #fit the model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse = mse / np.var(y_test) if np.var(y_test) != 0 else mse # normalize mse 
    fitted = pd.Series(y_pred, index = y_test.index)
    coefficents = pd.Series(model.coef_, index = X_train.columns)
    r2= model.score(X_test, y_test)
    sign_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))

    
    return {'mse': mse,
            'fitted': fitted,
            'actual': y_test,
            'coefficients': coefficents,
            'r2': r2,
            'sign_accuracy': sign_accuracy,}

def regression_per_symbol(df: pd.DataFrame,
                               date_col: str = 'DATETIME',
                               target: str = 'LOG_RETURN_NoOVERNIGHT',
                               features: list = None,
                               alpha: float = 0.1,
                               model: str = 'linear',
                               standardize: bool = False,
                               pt: float = 0.8,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Run regression per symbol and return results as a DataFrame.
    """
    results = []

    symbols = df['SYMBOL'].dropna().unique()

    for symbol in symbols:
        df_symbol = df[df['SYMBOL'] == symbol]
        
        # Skip if not enough data
        if df_symbol.shape[0] < 10:
            continue

        result = regression(df=df_symbol,
                            date_col=date_col,
                            target=target,
                            features=features,
                            alpha=alpha,
                            model=model,
                            standardize=standardize,
                            pt=pt)

        if verbose:
            print(f"{symbol}: MSE={result['mse']:.6f}, R²={result['r2']:.4f}, "
                  f"Sign Accuracy={result['sign_accuracy']:.2%}, "
                  f"Coefficients={result['coefficients'].to_dict()}")

        results.append({
            'symbol': symbol,
            'mse': result['mse'],
            'r2': result['r2'],
            'sign_accuracy': result['sign_accuracy'],
            'coefficients': result['coefficients'].to_dict()
        })

    return pd.DataFrame(results)

def plot_scores(results_df: pd.DataFrame, name: str = "OLS")-> None:
    """ Plot MSE, R², Sign Accuracy, Coefficients values for each symbol."""
    df = results_df.copy()

    # Plot MSE
    plt.figure(figsize=(12, 6))
    plt.bar(df['symbol'], df['mse'], color='lightcoral', edgecolor='black')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=90)
    plt.ylabel(" Mean Squared Error (MSE)")
    plt.title(f"RNormalized MSE per Symbol — {name}")
    plt.tight_layout()
    plt.show()

    # Plot R²
    plt.figure(figsize= (12, 6))
    plt.bar(df['symbol'], df['r2'], color='skyblue', edgecolor='black')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=90)
    plt.ylabel("R²")
    plt.title(f"R² per Symbol — {name}")
    plt.tight_layout()
    plt.show()

    # Plot Sign Accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(df['symbol'], df['sign_accuracy'], color='lightgreen', edgecolor='black')
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=90)
    plt.ylabel("Sign Accuracy")
    plt.title(f"SIgn Accuracy per Symbol — {name}")
    plt.tight_layout()
    plt.show()

    # Plot Coefficients
    coeff_df = pd.DataFrame(df['coefficients'].tolist(), index=df['symbol'])

    for col in coeff_df.columns:
            plt.figure(figsize=(12, 4))
            plt.bar(coeff_df.index, coeff_df[col], color='orange', edgecolor='black')
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.xticks(rotation=90)
            plt.ylabel("Coefficient Value")
            plt.title(f"Coefficient: {col} — {name}")
            plt.tight_layout()
            plt.show()


# Results

if __name__ == "__main__":

    # parameters
    BASE_DIR = Path.cwd()
    DATA_PATH = ".." / BASE_DIR / "data" / "processed" / "high_10m.parquet"
    NUM_LAGS = 4 #TODO: choose number of lags
    
    # Load data
    df = pd.read_parquet(DATA_PATH)
    df = df[df['LOG_RETURN_NoOVERNIGHT'] != 0]

    # Updating the df: Create DATETIME colum and lag features
    df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
    df = create_lag_features(df, NUM_LAGS, 'LOG_RETURN_NoOVERNIGHT')

    # Define features for regressiona
    features = [f'LOG_RETURN_NoOVERNIGHT_lag_{i}' for i in range(1, NUM_LAGS + 1)]
    
    # Run analysis for all symbols
    # 1. OLS regression
    filtered_df = df[df['SYMBOL'].isin(['ABIO', 'ZYXI'])] #TODO: REMOVE IT, JUST TO TEST WITH FEW SYMBOLS
    OLS_results_df = regression_per_symbol(filtered_df, date_col='DATETIME', target='LOG_RETURN_NoOVERNIGHT', features=features, alpha=0.1,  model='linear', standardize=False, pt=0.8)
    plot_scores(OLS_results_df, name="OLS")
    
    # 2. Ridge regression
    ridge_results_df = regression_per_symbol(filtered_df, date_col='DATETIME', target='LOG_RETURN_NoOVERNIGHT', features=features, alpha=0.1, model='ridge', standardize=False, pt=0.8)
    plot_scores(ridge_results_df, name="Ridge")

    # 3. Lasso regression
    lasso_results_df = regression_per_symbol(filtered_df, date_col='DATETIME', target='LOG_RETURN_NoOVERNIGHT', features=features, alpha=0.1, model='lasso', standardize=False, pt=0.8)
    plot_scores(lasso_results_df, name="Lasso")