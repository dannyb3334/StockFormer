import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

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

def fetch_and_clean(tickers):
    """
    Fetch and clean stock data for the given tickers.
    """
    data = yf.download(tickers, interval='1d', start='2018-01-01', end='2024-03-30') # Example date range

    # Change second level columns to their tickers index for easier processing
    data.columns = pd.MultiIndex.from_tuples(
        [(_, tickers.index(ticker)) for _, ticker in data.columns]
    )

    # Remove tickers with missing data
    for ticker_index, ticker in enumerate(tickers[:]):
        if data[('Open', ticker_index)].isna().any():
            tickers.remove(ticker)
            data = data.drop(ticker, axis=1, level=1)

    # Clean the data by removing outliers (values more than 3 std from mean)
    for col in data.columns:
        series = data[col]
        mean = series.mean()
        std = series.std()
        outliers = (series - mean).abs() > 3 * std
        data.loc[outliers, col] = float('nan')
    data.ffill(inplace=True)  # Forward fill to handle NaN values

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
    for ticker_index, ticker in enumerate(tickers):
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
        return_rate = data[('Close', ticker_index)].pct_change().fillna(0)
        new_cols['Y_RETURN_RATE'] = return_rate
        # Trend direction: 1 if return > 0 else -1
        new_cols['Y_TREND_DIRECTION'] = np.where(return_rate > 0, 1, -1)

        # Generate lagged features for each column (Open, High, Low, Close, Volume, Vwap) for 60 days
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Vwap']
        for i in range(1, 61):
            for col in cols:
                col_name = f'{col.upper()}{i}'
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
    data = data.stack(level=1).reset_index().rename(columns={'level_1': 'Ticker'})
    data.columns.name = None
    # Remove initial rows with insufficient lagged data
    data = data.iloc[num_tickers * 60:]
    data = data.reset_index(drop=True)

    # Neutralize all FEATURE columns by industry and market cap
    feature_cols = [col for col in data.columns if str(col).startswith('FEATURE')]
    for col in feature_cols:
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
    data.drop(columns=['Date'], inplace=True)
    data['Time_Slot'] = np.repeat(time_slots.values, num_tickers)
    print(data)
    
    return data

def create_period_splits(data, num_tickers, seq_splits_length, period_step, lag, lead):
    """
    Create period splits for training, validation, and test sets.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        num_tickers (int): Number of tickers.
        seq_splits_length (int): Length of each sequence split.
        period_step (int): Step size for each period.
        lag (int): Number of lagged time steps.
        lead (int): Number of lead time steps.
    Returns:
        dict: Dictionary containing period splits.
    """
    # Get locations of target columns
    target_cols_locs = np.asarray([data.columns.get_loc('Y_RETURN_RATE'), data.columns.get_loc('Y_TREND_DIRECTION')])
    # Remove Time_Slot column from standardization
    time_slot = data['Time_Slot']
    time_slot = time_slot.to_numpy()
    data.drop(columns=['Time_Slot'], inplace=True)

    # Create period splits
    period_start = 0
    period_count = 0
    period_splits = {}
    period_step = period_step * num_tickers
    
    seq_splits_length = seq_splits_length * num_tickers
    for period_end in range(seq_splits_length, len(data), period_step):
        # Create a slice for the current period
        period_slice = data[period_start:period_end].copy()
        date_slice = time_slot[period_start:period_end][::num_tickers]
        ticker_target_stats = {}
        for ticker_index in range(num_tickers):
            # Standardize tickers separately 
            ticker_rows = period_slice[ticker_index::num_tickers]
            mean = np.mean(ticker_rows)
            std = np.std(ticker_rows)
            period_slice[ticker_index::num_tickers] = (ticker_rows - mean) / std
            ticker_target_stats[ticker_index] = {'mean': mean, 'std': std}
            ticker_target_stats[ticker_index] = {'mean': mean, 'std': std}

        # Create sequences for model input (lag for features, lead for targets)
        Xs = [] # Input sequences
        Ts = [] # Timestamps
        Ys = [] # Target sequences

        # Group every num_tickers rows into arrays
        grouped_by_date = period_slice.reshape(-1, num_tickers, period_slice.shape[1])
        for i in range(len(grouped_by_date) - lag - lead):
            Xs.append(grouped_by_date[i:i + lag])
            Ys.append(grouped_by_date[i + lag:i + lag + lead, :, target_cols_locs])
            Ts.append(date_slice[i:i + lag])  # timestamp for the last lag day

        # Convert lists to numpy arrays
        Xs = np.asarray(Xs)
        Ts = np.asarray(Ts)
        Ys = np.asarray(Ys)

        # Organize the data into a dictionary
        len_Xs = len(Xs)
        training_size = int(len_Xs * 0.75)
        val_test_size = int(len_Xs * 0.125)
        # Split into training, validation, and test sets
        period_splits[period_count] = {
            'training': { 'X': Xs[:training_size], 'Y': Ys[:training_size], 'Ts': Ts[:training_size] },
            'validation': { 'X': Xs[training_size:training_size + val_test_size], 'Y': Ys[training_size:training_size + val_test_size], 'Ts': Ts[training_size:training_size + val_test_size] },
            'test': { 'X': Xs[training_size + val_test_size:], 'Y': Ys[training_size + val_test_size:], 'Ts': Ts[training_size + val_test_size:] },
            'target_standardization': ticker_target_stats
        }

        period_count += 1
        period_start += period_step
        
    return period_splits

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
    
    ## Download stock data for multiple tickers with daily intervals
    #url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    #
    ## Read the HTML table into a DataFrame
    #table = pd.read_html(url)[0]
#
    ## Extract the list of tickers from the 'Symbol' column
    #tickers = table['Symbol'].tolist()
    tickers = ['AAPL', 'MSFT']

    # Download and clean the data
    data, tickers = fetch_and_clean(tickers)

    # Create price and volume features
    data = create_features(data, tickers)
    num_tickers = len(tickers)

    # Create period splits for training, validation, and test sets
    seq_splits_length = 486+81+81
    period_step = 81
    lag = 20
    lead = 2
    period_splits = create_period_splits(data, num_tickers, seq_splits_length, period_step, lag, lead)

    # Save the period_splits dictionary to a file
    save_data(period_splits, 'period_splits.pkl')
