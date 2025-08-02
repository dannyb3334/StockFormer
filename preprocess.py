import os
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import yaml
from tqdm import tqdm

def neutralize_factor(factor_series, industry_series, market_value_series):
    """
    Neutralize a factor by regressing it against industry and market value.
    """
    # Create dummy variables for industry
    industry_dummies = pd.get_dummies(industry_series, prefix='industry')
    # Concatenate industry dummies and market value into feature matrix X
    X = pd.concat([industry_dummies, market_value_series.rename('mkt_val')], axis=1)
    X = sm.add_constant(X)  # Add constant term for regression
    # Find valid indices where both factor and X are not NaN
    valid_idx = factor_series.dropna().index.intersection(X.dropna().index)
    y = factor_series.loc[valid_idx].astype(float)
    X = X.loc[valid_idx].astype(float)
    # Fit OLS regression model
    model = sm.OLS(y, X).fit()
    # Return residuals (neutralized factor)
    return y - model.predict(X)

def fetch_and_clean(tickers, start_date=None, end_date=None, day_range=None):
    """
    Fetch and clean stock data for the given tickers.
    """
    try:
        if day_range:
            data = yf.download(tickers, interval='1d', period=f"{day_range}d", auto_adjust=True)
        else:
            data = yf.download(tickers, interval='1d', start=start_date, end=end_date, auto_adjust=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame(), []

    # Handle case where no data is returned
    if data.empty:
        print("No data downloaded for any tickers")
        return pd.DataFrame(), []

    # Change second level columns to their tickers index for easier processing
    data.columns = pd.MultiIndex.from_tuples(
        [(_, tickers.index(ticker)) for _, ticker in data.columns]
    )

    # Identify tickers with missing data
    valid_tickers = []
    valid_ticker_indices = []
    
    for ticker_index, ticker in enumerate(tickers):
        try:
            if not data[('Open', ticker_index)].isna().any():
                valid_tickers.append(ticker)
                valid_ticker_indices.append(ticker_index)
            else:
                print(f"Removing ticker {ticker} due to missing data")
        except KeyError:
            print(f"Removing ticker {ticker} - not found in downloaded data")
    
    # Keep only valid tickers and reindex columns
    if valid_ticker_indices:
        # Create new data with only valid tickers and sequential indices using pd.concat
        columns_to_concat = {}
        for new_idx, old_idx in enumerate(valid_ticker_indices):
            for col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if (col_name, old_idx) in data.columns:
                    columns_to_concat[(col_name, new_idx)] = data[(col_name, old_idx)]
        
        # Use pd.concat for better performance
        data = pd.concat(columns_to_concat, axis=1)
        tickers = valid_tickers
    else:
        print("No valid tickers found")
        return pd.DataFrame(), []

    # Clean the data by removing outliers (values more than 3 std from mean)
    for col in data.columns:
        series = data[col]
        mean = series.mean()
        std = series.std()
        outliers = (series - mean).abs() > 3 * std
        data.loc[outliers, col] = float('nan')
    data.ffill(inplace=True)  # Forward fill to handle NaN values
    data.bfill(inplace=True)  # Backward fill to handle any remaining NaN values
    return data, tickers

def create_features(data, tickers):
    """
    Create price, volume, and time features for the given tickers.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        tickers (list): List of stock tickers.
    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    num_tickers = len(tickers)
    # Create feature columns for each ticker
    industry_names = {} # Dictionary to store industry name indices corresponding to their names
    idx_industry_names = 0
    for ticker_index, ticker in enumerate(tqdm(tickers, desc="Feature Engineering")):
        t = yf.Ticker(ticker)
        new_cols = {}
        # Assign industry index if not already present
        industry_names.setdefault(t.info['industry'], idx_industry_names)
        idx_industry_names += 1
        new_cols['Industry'] = industry_names[t.info['industry']]
        # Calculate Market Cap as shares outstanding * close price
        new_cols['Market_Cap'] = t.info['sharesOutstanding'] * data[('Close', ticker_index)].values
        # Calculate VWAP (Volume Weighted Average Price) (Additional feature)
        data[('Vwap', ticker_index)] = ((data[('Close', ticker_index)].values + data[('High', ticker_index)].values + 
                                        data[('Low', ticker_index)].values) / 3 * data[('Volume', ticker_index)].values).cumsum() / \
                                            data[('Volume', ticker_index)].values.cumsum()
        # Calculate interval return rate
        return_rate = data[('Close', ticker_index)].pct_change().shift(-1).fillna(0)
        new_cols['Y_RETURN_RATE'] = return_rate
        # Trend direction: 1 if return > 0 else -1
        new_cols['Y_TREND_DIRECTION'] = np.where(return_rate > 0, 1, -1)

        # Generate lagged features for each column (Open, High, Low, Close, Volume, Vwap) for 60 days
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Vwap']

        for i in range(0, 60):
            for col in cols:
                if col == 'Volume':
                    # Lagged volume feature normalized by current volume
                    feature = data[('Volume', ticker_index)].shift(i) / (data[('Volume', ticker_index)] + 1e-12)
                else:
                    # Lagged price features normalized by current close price
                    feature = data[(col, ticker_index)].shift(i) / data[('Close', ticker_index)]
                new_cols[f'FEATURE_{col.upper()}{i}'] = feature

        # Concatenate new_cols as DataFrame to data for this ticker
        new_features = pd.DataFrame(new_cols, index=data.index)
        # Add new features to main data DataFrame
        data = pd.concat([data, pd.concat({(col, ticker_index): new_features[col] for col in new_features.columns}, axis=1)], axis=1)

    # Move ticker index to a column and flatten columns to single level
    data = data.stack(level=1, future_stack=True).reset_index().rename(columns={'level_1': 'Ticker'})
    data.columns.name = None
    # Remove initial rows with insufficient lagged data
    data = data.iloc[num_tickers * 60:]
    data = data.reset_index(drop=True)

    # Neutralize all FEATURE columns by industry and market cap
    feature_cols = [col for col in data.columns if str(col).startswith('FEATURE')]
    for col in tqdm(feature_cols, desc="Neutralizing Features"):
        data[col] = neutralize_factor(
            data[col],
            data['Industry'],
            data['Market_Cap']
        )
    # Drop unnecessary columns
    data.drop(columns=['Industry', 'Market_Cap', 'Close', 'High', 'Low', 'Open', 'Volume', 'Vwap', 'Ticker'], inplace=True)

    # Create time slots
    time_slots = data['Date'][::num_tickers]
    time_slots = time_slots.reset_index(drop=True)
    offset = time_slots.iloc[:252]

    # Find the index of the date closest to the start of the year
    first_day_of_year = pd.to_datetime([f"{d.year}-01-01" for d in offset])
    diffs = (offset - first_day_of_year).abs()
    closest_idx = diffs.idxmin()
    time_slots = ((time_slots.index + (252 - closest_idx)) % (252)).astype(int)

    # Replace 'Date' column with time slots
    data = data.drop(columns=['Date']).copy()
    data['Time_Slot'] = np.repeat(time_slots.values, num_tickers)
    
    return data

def create_predict_data(tickers, lag, standard_window):
    """
    Prepare model input data for prediction using the latest available data.
    Downloads, cleans, and feature-engineers the data, then standardizes features per ticker.
    Returns lagged feature sequences and timestamps for prediction.
    Args:
        tickers (list): List of stock tickers.
        lag (int): Number of lagged time steps for model input.
        standard_window (int): Number of days to consider for standardization.
    Returns:
        tuple: Tuple containing:
            - Xs (np.ndarray): Input sequences of shape (num_samples, lag, num_tickers, num_features).
            - Ts (np.ndarray): Timestamps corresponding to the last lag
    """
    # Download and clean the data
    data, tickers = fetch_and_clean(tickers, day_range=standard_window)

    # Create price and volume features
    data = create_features(data, tickers)
    num_tickers = len(tickers)

    # Get locations of factor columns
    factor_cols_locs = np.asarray([data.columns.get_loc(col) for col in data.columns if str(col).startswith('FEATURE')])

    # Remove Time_Slot column from standardization
    time_slot = data['Time_Slot']
    time_slot = time_slot.to_numpy()
    data.drop(columns=['Time_Slot'], inplace=True)

    # Standardize features per ticker
    data_copy = data.copy()
    date_slice = time_slot[::num_tickers]
    for ticker_index in range(num_tickers):
        # Standardize features for each ticker separately
        ticker_rows = data_copy.iloc[ticker_index::num_tickers, factor_cols_locs]
        mean = ticker_rows.mean(axis=0)
        std = ticker_rows.std(axis=0)
        standardized_rows = (ticker_rows - mean) / (std + 1e-12)  # Avoid division by zero
        data_copy.iloc[ticker_index::num_tickers, factor_cols_locs] = standardized_rows

    # Create lagged feature sequences and timestamps for prediction
    Xs = [] # Input sequences
    Ts = [] # Timestamps

    # Group every num_tickers rows into arrays
    period_slice_values = data_copy.values  # Convert to numpy array
    grouped_by_date = period_slice_values.reshape(-1, num_tickers, data_copy.shape[1])
    for i in range(len(grouped_by_date) - lag):
        Xs.append(grouped_by_date[i:i + lag])
        Ts.append(date_slice[i:i + lag])  # timestamp for the last lag day

    # Convert lists to numpy arrays
    Xs = np.asarray(Xs)
    Ts = np.asarray(Ts)

    return Xs, Ts


def create_period_splits(data, tickers, seq_splits_length, period_step, lag, lead, train_split, val_split):
    """
    Create rolling period splits for training, validation, and test sets for time series modeling.
    Each period is standardized per ticker, then split into lagged feature/target sequences.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        tickers (list): List of ticker symbols.
        seq_splits_length (int): Length of each sequence split (in time steps).
        period_step (int): Step size for each period (in time steps).
        lag (int): Number of lagged time steps for model input.
        lead (int): Number of lead time steps for model target.
        train_split (float): Fraction of samples for training set.
        val_split (float): Fraction of samples for validation set.
    Returns:
        dict: Dictionary containing period splits with training, validation, and test sets.
    """
    # Get locations of target columns and factor columns
    target_cols_locs = np.asarray([data.columns.get_loc('Y_RETURN_RATE'), data.columns.get_loc('Y_TREND_DIRECTION')])
    factor_cols_locs = np.asarray([data.columns.get_loc(col) for col in data.columns if str(col).startswith('FEATURE')])

    # Remove Time_Slot column from standardization
    time_slot = data['Time_Slot']
    time_slot = time_slot.to_numpy()
    data.drop(columns=['Time_Slot'], inplace=True)

    # Create rolling period splits
    period_start = 0
    period_count = 0
    num_tickers = len(tickers)
    period_step *= num_tickers
    seq_splits_length *= num_tickers

    for period_end in tqdm(range(seq_splits_length, len(data), period_step), desc="Creating Period Splits"):
        # Create a slice for the current period
        period_slice = data[period_start:period_end].copy()
        date_slice = time_slot[period_start:period_end][::num_tickers]
        # Standardize features for each ticker in the period
        for ticker_index in range(num_tickers):
            ticker_rows = period_slice.iloc[ticker_index::num_tickers, factor_cols_locs]
            mean = ticker_rows.mean(axis=0)
            std = ticker_rows.std(axis=0)
            standardized_rows = (ticker_rows - mean) / (std + 1e-12)  # Avoid division by zero
            period_slice.iloc[ticker_index::num_tickers, factor_cols_locs] = standardized_rows

        # Create lagged feature/target sequences and timestamps for this period
        Xs = [] # Input sequences
        Ts = [] # Timestamps
        Ys = [] # Target sequences

        # Group every num_tickers rows into arrays
        period_slice_values = period_slice.values  # Convert to numpy array
        grouped_by_date = period_slice_values.reshape(-1, num_tickers, period_slice.shape[1])
        for i in range(len(grouped_by_date) - lag - lead):
            Xs.append(grouped_by_date[i:i + lag])
            Ys.append(grouped_by_date[i + lag:i + lag + lead, :, target_cols_locs])
            Ts.append(date_slice[i:i + lag])  # timestamp for the last lag day

        # Convert lists to numpy arrays
        Xs = np.asarray(Xs)
        Ts = np.asarray(Ts)
        Ys = np.asarray(Ys)

        # Organize the data into a dictionary with train/val/test splits
        len_Xs = len(Xs)
        training_size = int(len_Xs * train_split)
        val_test_size = int(len_Xs * val_split)

        period_splits = {
            'training': { 'X': Xs[:training_size], 'Y': Ys[:training_size], 'Ts': Ts[:training_size] },
            'validation': { 'X': Xs[training_size:training_size + val_test_size], 'Y': Ys[training_size:training_size + val_test_size], 'Ts': Ts[training_size:training_size + val_test_size] },
            'test': { 'X': Xs[training_size + val_test_size:], 'Y': Ys[training_size + val_test_size:], 'Ts': Ts[training_size + val_test_size:] },
        }

        export = {'data': period_splits, 'tickers': tickers, 'seq_len': lag, 'pred_len': lead }

        save_data(export, os.path.join('training_periods', f'period_split_{period_count}.pkl'))

        del export, period_splits, Xs, Ys, Ts, period_slice_values, grouped_by_date

        period_count += 1
        period_start += period_step

def save_data(data, filename):
    """
    Save the period_splits dictionary to a file using pickle.
    Args:
        data (dict): Dictionary to save.
        filename (str): Filename to save the data.
    """
    # Save the period_splits dictionary to a file using pickle
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    # Load parameters from YAML file
    with open('config.yaml', 'r') as f:
        params = yaml.safe_load(f)

    train_params = params.get('train_params')
    model_params = params.get('model_params')
    tickers = model_params.get('tickers')
    period_len = train_params.get('period_len')
    period_step = model_params.get('min_len_for_pred')
    lag = model_params.get('seq_len')
    lead = model_params.get('pred_len')
    train_start_date = model_params.get('trained_period').get('start')
    train_end_date = model_params.get('trained_period').get('end')
    train_split = train_params.get('train_split')
    val_split = train_params.get('val_split')

    print("Processing training data...")
    print(" ".join(tickers))

    # Download and clean the data
    data, tickers = fetch_and_clean(tickers, start_date=train_start_date, end_date=train_end_date)

    # Create price and volume features
    data = create_features(data, tickers)

    # Create period splits for training, validation, and test sets
    create_period_splits(data, tickers, period_len, period_step, lag, lead, train_split, val_split)

    # Print summary    
    print(f"Preprocessing completed successfully!")
    print(f"Processed {len(tickers)} tickers: {tickers}")
    #print(f"Created {len(period_data['period_splits'])} period splits")
    print(f"Saved data to 'period_splits.pkl'")

    # Update config.yaml with new valid tickers
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['model_params']['tickers'] = tickers

    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    print("Updated config.yaml with valid tickers.")
    
