import requests
import pandas as pd
import numpy as np 
import talib as ta
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
import sqlite3
import pyarrow.parquet as pq 
import pyarrow as pa
import os
def fetch_data_from_quandl(symbol_list, api_key='9rrbTAirnbfgjYJ_QziJ'):
    all_data = {}
    for symbol in symbol_list:
        data = fetch_data_for_symbol(symbol, api_key)
        if data is not None:
            all_data[symbol] = data
    return all_data

def fetch_data_for_symbol(symbol, api_key):
    try:
        base_url = f'https://www.quandl.com/api/v3/datasets/WIKI/{symbol}.json'
        params = {
            'api_key': api_key,
        }

        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Error: Failed to fetch data for {symbol}. Status code: {response.status_code}")
            return None

        data = response.json()

        if 'quandl_error' in data:
            print(f"Error: {data['quandl_error']['message']}")
            return None

        df = pd.DataFrame(data['dataset']['data'], columns=data['dataset']['column_names'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def prompt_for_symbols():
    symbol_input = input("Enter one or more stock symbols separated by commas: ")
    return [symbol.strip().upper() for symbol in symbol_input.split(',')]

def choose_indicators():
    print("Choose from the following indicators:")
    print("1. RSI")
    print("2. OBV")
    print("3. Parabolic SAR")
    print("4. SMA")
    selected_indicators = input("Enter the indicator numbers separated by commas (e.g., 1,2,3): ")
    indicator_map = {
        '1': 'RSI',
        '2': 'OBV',
        '3': 'SAR',
        '4': 'SMA'
    }
    indicators = []
    for indicator in selected_indicators.split(','):
        if indicator in indicator_map:
            indicators.append(indicator_map[indicator])
        else:
            print(f"Invalid indicator number: {indicator}")
    return indicators

def generate_features(df, selected_indicators):
    for indicator in selected_indicators:
        if indicator.upper() == 'OBV':
            df['OBV'] = ta.OBV(df['Close'], df['Volume'])
        elif indicator.upper() == 'SAR':
            df['SAR'] = ta.SAR(df['High'], df['Low'])
        elif indicator.upper() == 'RSI':
            df['RSI'] = ta.RSI(df['Close'])
        elif indicator.upper() == 'SMA':
            df['SMA'] = ta.SMA(df['Close'])
        else:    
            try:
                indicator_function = getattr(ta, indicator)
                indicator_outputs = indicator_function(df['Open'], df['High'], df['Low'], df['Close'], df['Volume'])
                if isinstance(indicator_outputs, tuple):
                    for i, output in enumerate(indicator_outputs):
                        df[f"{indicator}_{i+1}"] = output
                else:
                    df[indicator] = indicator_outputs
            except AttributeError:
                print(f"Indicator '{indicator}' not found in TA library")

""" def export_to_csv(data):
    for symbol, df in data.items():
        df.to_csv(f'{symbol}_data.csv')

def export_to_excel(data):
    try:
        with pd.ExcelWriter('stock_data.xlsx') as writer:
            for symbol, df in data.items():
             df.to_excel(writer, sheet_name=symbol)
        print("Data exported to excel file.")
    except PermissionError:
        print("Permission Denied") """

def impute_missing_values_with_lrgc(df, n_components=5, scale=True, random_state=42):
    if df.isnull().any().any():
        print("Missing values detected. Imputing...")
        if scale:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        else:
            df_scaled = df.copy()

        numerical_cols = df.select_dtypes(include='number').columns
        pca = PCA()
        df_filled = df.copy()
        for col in numerical_cols:
            missing_mask = df[col].isnull()
            if missing_mask.sum() > 0:
                observed_data = df[col][~missing_mask].values.reshape(-1, 1)
                pca.fit(observed_data)
                predicted_data = pca.transform(observed_data)
                imputed_latent_data = np.random.multivariate_normal(mean=np.zeros(predicted_data.shape[1]),
                                                                     cov=np.eye(predicted_data.shape[1]),
                                                                     size=missing_mask.sum())
                imputed_data = pca.inverse_transform(imputed_latent_data)
                df_filled.loc[missing_mask, col] = imputed_data[:, 0]
        if scale:
            df_filled = pd.DataFrame(scaler.inverse_transform(df_filled), columns=df.columns, index=df.index)
        print("Imputation complete.")
        return df_filled
    else:
        print("No missing values detected.")
        return df


def detect_outliers(df, threshold=3):
    z_scores = ((df - df.mean()) / df.std()).abs()
    outliers_mask = (z_scores>threshold)
    df_corrected=df.mask(outliers_mask)
    return df_corrected

def plot_data_with_indicators(df,selected_indicators):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close', color='blue')

    for indicator in selected_indicators:
        plt.plot(df.index, df[indicator], label=indicator)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('OHLC Data with Indicators')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_rolling_stats(df, window=20):
    df['rolling_mean'] = df['Close'].rolling(window=window).mean()
    df['rolling_std'] = df['Close'].rolling(window=window).std()

def mean_reversion_signal(df):
    df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']
    df['mean_reversion_signal'] = np.where(df['z_score'] > 1.0, 'Sell', np.where(df['z_score'] < -1.0, 'Buy', 'Hold'))

def resample_data(df, frequency='D'):
    df.index = pd.to_datetime(df.index)
    resampled_df = df.resample(frequency).agg({
        'Open': 'first',    
        'High': 'max',      
        'Low': 'min',       
        'Close': 'last',    
        'Volume': 'sum'     
    })

    return resampled_df

def connect_to_database(database_name='stock_data.db'):
    conn = sqlite3.connect(database_name)
    return conn

def create_tables(conn, data):
    c = conn.cursor()
    for symbol, df in data.items():
        df.to_sql(symbol, conn, if_exists='replace')
    conn.commit()

def insert_data(conn, table_name, data):
    data.to_sql(table_name, conn, if_exists='replace')

def save_to_database(data):
    conn = connect_to_database()
    create_tables(conn, data)
    conn.close()
    print("Data saved to SQLite database.")

def export_to_parquet(data, compression='snappy'):
    for symbol, df in data.items():
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'{symbol}_data.parquet', compression=compression)
        print(f"Data for {symbol} exported to Parquet file with {compression} compression.")

def load_parquet_data(directory):
    data = {}
    print(f"Loading parquet data from directory: {directory}")
    for file in os.listdir(directory):
        if file.endswith(".parquet"):
            print(f"Found parquet file: {file}")
            symbol = file.split("_")[0]
            print(f"Loading parquet file for symbol: {symbol}")
            table = pq.read_table(os.path.join(directory, file))
            data[symbol] = table.to_pandas()
    return data

def main():
    symbol_list = prompt_for_symbols()
    data = fetch_data_from_quandl(symbol_list)

    parquet_directory = 'C:\\Invsto Project'
    parquet_data = load_parquet_data(parquet_directory)

    if parquet_data:
        print("Parquet data loaded successfully:")
        for symbol, df in parquet_data.items():
            print(f"\nData for {symbol}:")
            print(df)
    else:
        print("No parquet data found.")

    if data:
        print("Data fetched successfully:")
        for symbol, df in data.items():
            print(f"\nData for {symbol}:")
            print(df)

        selected_indicators = choose_indicators()
        print("Selected indicators:", selected_indicators)

        trained_models = {}

        for symbol , df in data.items():
            generate_features(df, selected_indicators)
            print(f"\nFeatures for {symbol}:")
            print(df)

            X = df[selected_indicators].values
            y = df['Close'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            trained_models[symbol] = model

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"\nTraining Score: {train_score}")
            print(f"Test Score: {test_score}")

        for symbol, df in data.items():
            X_pred = df[selected_indicators].values
            predicted_prices = trained_models[symbol].predict(X_pred)
            df['Predicted_Close'] = predicted_prices

            print(f"\nPredictions for {symbol}:")
            print(df[['Close', 'Predicted_Close']])

            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['Close'], label='Actual Close', color='blue')
            plt.plot(df.index, df['Predicted_Close'], label='Predicted Close', color='red')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Actual vs. Predicted Close Prices for {symbol}')
            plt.legend()
            plt.grid(True)
            plt.show()

        for symbol, df in data.items():
            generate_features(df,selected_indicators)
            calculate_rolling_stats(df)
            mean_reversion_signal(df)

            print(f"\nFeatures for {symbol}:")
            print(df[['Close', 'rolling_mean', 'rolling_std', 'z_score', 'mean_reversion_signal']])
        save_to_database(data)
    else:
        print("Failed to fetch data for any symbol.")
if __name__ == "__main__":
    main()