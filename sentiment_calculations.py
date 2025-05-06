#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def load_sentiment_data(filepath='finsen.csv', date_column='Time'):
    """
    Loads the sentiment dataset (e.g., finsen) from a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        date_column (str): The name of the column containing dates
                           (expected format: day/month/year).

    Returns:
        pandas.DataFrame: DataFrame with sentiment data and parsed dates,
                          indexed by date, or None if loading fails.
    """
    print(f"Attempting to load sentiment data from: {filepath}")
    try:
        
        df = pd.read_csv(filepath, parse_dates=[date_column], dayfirst=True)
        df = df.set_index(date_column)
        df.sort_index(inplace=True)

        print("Sentiment data loaded successfully.")
        print(f"Columns available: {df.columns.tolist()}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except KeyError:
        print(f"Error: Date column '{date_column}' not found in the CSV file.")
        print(f"Available columns: {pd.read_csv(filepath, nrows=0).columns.tolist()}")
        return None
    except Exception as e:
        print(f"An error occurred while loading sentiment data: {e}")
        return None

finsen_filepath = 'finsen.csv'
date_col_name = 'Time'

sentiment_df = load_sentiment_data(filepath=finsen_filepath, date_column=date_col_name)

if sentiment_df is not None:
    print("\nSentiment DataFrame Info:")
    sentiment_df.info()
    print("\nFirst 5 rows of sentiment data:")
    print(sentiment_df.head())
else:
    print("\nFailed to load sentiment data. Please check the filepath and date column name.")
    
def load_afinn_lexicon(filepath):
    """Loads the AFINN lexicon from a file (word<tab>score format)."""
    print(f"Loading AFINN lexicon from: {filepath}")
    afinn = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    afinn[parts[0]] = int(parts[1])
        if not afinn:
             print("Warning: AFINN lexicon loaded but is empty. Check file format.")
        else:
             print(f"AFINN lexicon loaded successfully with {len(afinn)} entries.")
        return afinn
    except FileNotFoundError:
        print(f"Error: AFINN lexicon file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading AFINN lexicon: {e}")
        return None

def calculate_text_sentiment(text, afinn_lexicon):
    """Calculates the average sentiment score for a text string."""
    if not isinstance(text, str) or not afinn_lexicon:
        return 0 

    words = re.findall(r'\b\w+\b', text.lower())
    score = 0
    words_found = 0
    for word in words:
        if word in afinn_lexicon:
            score += afinn_lexicon[word]
            words_found += 1

    return score / words_found if words_found > 0 else 0

def calculate_daily_sentiment(sentiment_data_df, afinn_lexicon, column_to_analyze='title'):
    """
    Calculates daily average sentiment scores from a DataFrame of articles.

    Args:
        sentiment_data_df (pd.DataFrame): DataFrame loaded from finsen data, indexed by date.
        afinn_lexicon (dict): Loaded AFINN lexicon {word: score}.
        column_to_analyze (str): Column name containing text ('title' or 'content').

    Returns:
        pd.Series: Series containing average sentiment score per day, indexed by date, or None.
    """
    if afinn_lexicon is None:
        print("Error: Cannot calculate sentiment without a loaded AFINN lexicon.")
        return None
    if column_to_analyze not in sentiment_data_df.columns:
        print(f"Error: Column '{column_to_analyze}' not found in sentiment DataFrame.")
        return None

    print(f"Calculating sentiment scores based on '{column_to_analyze}' column...")
    sentiment_data_df['article_sentiment'] = sentiment_data_df[column_to_analyze].apply(
        lambda text: calculate_text_sentiment(text, afinn_lexicon)
    )

    print("Aggregating scores to daily average...")
    daily_sentiment_scores = sentiment_data_df['article_sentiment'].groupby(sentiment_data_df.index).mean()

    print("Daily sentiment calculation complete.")
    return daily_sentiment_scores


afinn_file_path = 'AFINN-en-165.txt'

if 'sentiment_df' in locals() and isinstance(sentiment_df, pd.DataFrame):
    
    afinn = load_afinn_lexicon(afinn_file_path)

    if afinn:
        text_column = 'Title'
        daily_sentiment = calculate_daily_sentiment(sentiment_df, afinn, column_to_analyze=text_column)

        if daily_sentiment is not None:
            print("\nDaily Sentiment Scores calculated successfully:")
            print(daily_sentiment.head())
            print(f"\nNumber of days with sentiment scores: {len(daily_sentiment)}")

        else:
            print("\nFailed to calculate daily sentiment scores.")
    else:
        print("\nCannot proceed without loading the AFINN lexicon.")

else:
    print("\nVariable 'sentiment_df' not found or is not a DataFrame.")
    print("Please ensure you have loaded the finsen data successfully")
    print("into a DataFrame named 'sentiment_df' in your local environment before running this.")
    
def align_sentiment_with_stock_data(stock_df, sentiment_series, start_date, end_date):
    """
    Filters stock and sentiment data for a period, merges them, and checks for missing sentiment.

    Args:
        stock_df (pd.DataFrame): DataFrame with stock data (prices, returns),
                                 indexed by date (DatetimeIndex).
        sentiment_series (pd.Series): Series with daily sentiment scores,
                                      indexed by date (DatetimeIndex).
        start_date (str): Start date for the test period (e.g., '2017-01-01').
        end_date (str): End date for the test period (e.g., '2019-01-01').

    Returns:
        pd.DataFrame: Merged DataFrame containing stock data and sentiment
                      for the test period, or None if inputs are invalid.
    """
    print(f"Aligning data for period: {start_date} to {end_date}")

    if not isinstance(stock_df.index, pd.DatetimeIndex):
        try:
            stock_df.index = pd.to_datetime(stock_df.index)
        except Exception as e:
            print(f"Error converting stock_df index to DatetimeIndex: {e}")
            return None
    if not isinstance(sentiment_series.index, pd.DatetimeIndex):
        try:
            sentiment_series.index = pd.to_datetime(sentiment_series.index)
        except Exception as e:
            print(f"Error converting sentiment_series index to DatetimeIndex: {e}")
            return None

    stock_test = stock_df[(stock_df.index >= start_date) & (stock_df.index <= end_date)].copy()
    if stock_test.empty:
        print(f"Warning: No stock data found for the period {start_date} to {end_date}.")
        return None
    print(f"Filtered stock data shape: {stock_test.shape}")

    sentiment_test = sentiment_series[(sentiment_series.index >= start_date) & (sentiment_series.index <= end_date)].copy()
    if sentiment_test.empty:
        print(f"Warning: No sentiment data found for the period {start_date} to {end_date}.")
    print(f"Filtered sentiment data shape: {len(sentiment_test)}")

    sentiment_test.name = 'Sentiment'
    merged_df = stock_test.join(sentiment_test, how='left')

    missing_sentiment_count = merged_df['Sentiment'].isnull().sum()
    total_days = len(merged_df)

    print(f"\nMerge complete. Total trading days in period: {total_days}")
    if missing_sentiment_count > 0:
        print(f"Found {missing_sentiment_count} trading days with missing sentiment scores.")
    else:
        print("No missing sentiment scores found for trading days in the period.")

    return merged_df


test_start_date = '2016-01-01'
test_end_date = '2020-01-01'

if ('stock_data_with_logret' in locals() and isinstance(stock_data_with_logret, pd.DataFrame) and
    'daily_sentiment' in locals() and isinstance(daily_sentiment, pd.Series)):

    test_data_aligned = align_sentiment_with_stock_data(
        stock_data_with_logret,
        daily_sentiment,
        test_start_date,
        test_end_date
    )

    if test_data_aligned is not None:
        print("\nAligned Test Data Info:")
        test_data_aligned.info()
        print("\nFirst 5 rows of aligned data:")
        print(test_data_aligned.head())
        print("\nLast 5 rows of aligned data:")
        print(test_data_aligned.tail())
        
    else:
        print("\nFailed to align data.")

else:
    print("\nRequired DataFrames/Series ('stock_data_with_logret', 'daily_sentiment') not found.")
    print("Please ensure data is loaded and prepared in your local environment before running this.")
    
def create_price_logret_features(price_df):
    """
    Calculates log returns from adjusted close prices and combines
    prices and log returns into a single DataFrame with MultiIndex columns.

    Args:
        price_df (pd.DataFrame): DataFrame with DatetimeIndex and columns
                                 representing adjusted close prices for symbols.

    Returns:
        pd.DataFrame: DataFrame with MultiIndex columns (StockSymbol, Feature)
                      containing 'AdjClose' and 'LogRet', or None if input is invalid.
    """
    if not isinstance(price_df.index, pd.DatetimeIndex):
        print("Error: Input DataFrame index must be a DatetimeIndex.")
        return None
    if price_df.empty:
        print("Error: Input DataFrame is empty.")
        return None

    print("Calculating log returns...")
    log_returns_df = np.log(price_df / price_df.shift(1))

    print("Combining prices and log returns...")
    price_df_multi = price_df.copy()
    price_df_multi.columns = pd.MultiIndex.from_product([price_df.columns, ['AdjClose']],
                                                        names=['StockSymbol', 'Feature'])

    log_returns_df.columns = pd.MultiIndex.from_product([log_returns_df.columns, ['LogRet']],
                                                        names=['StockSymbol', 'Feature'])

    combined_df = pd.concat([price_df_multi, log_returns_df], axis=1)
    combined_df = combined_df.sort_index(axis=1)

    print("Finished creating combined feature DataFrame.")
    return combined_df

if 'merged_df' in locals() and isinstance(merged_df, pd.DataFrame):

    stock_features_df = create_price_logret_features(merged_df)

    if stock_features_df is not None:
        print("\nCombined Features DataFrame Info:")
        stock_features_df.info()
        print("\nFirst 5 rows (showing initial NaN for LogRet):")
        print(stock_features_df.head())
        print("\n--- Ready for Alignment Step ---")
        test_start_date = '2016-01-01'
        test_end_date = '2020-01-01'

        if 'daily_sentiment' in locals() and isinstance(daily_sentiment, pd.Series):
            print(f"Now, you should call align_sentiment_with_stock_data using 'stock_features_df'")
            print(f"and 'daily_sentiment' for the period {test_start_date} to {test_end_date}.")
            
        else:
            print("Variable 'daily_sentiment' not found. Please calculate it first.")

    else:
        print("\nFailed to create combined features DataFrame.")

else:
    print("\nVariable 'merged_df' not found or is not a DataFrame.")
    print("Please ensure the stock price data is loaded into 'merged_df' first.")
    
def align_sentiment_with_stock_data(stock_df: pd.DataFrame, sentiment_series: pd.Series, start_date: str, end_date: str):
    """
    Filters stock and sentiment data for a period, merges them by adding sentiment
    as a new column, and checks for missing sentiment.

    Args:
        stock_df (pd.DataFrame): DataFrame with stock data (e.g., prices, returns),
                                 indexed by date (DatetimeIndex). Must contain MultiIndex columns.
        sentiment_series (pd.Series): Series with daily sentiment scores,
                                      indexed by date (DatetimeIndex).
        start_date (str): Start date for the test period (e.g., '2016-01-01').
        end_date (str): End date for the test period (e.g., '2023-01-01').

    Returns:
        pd.DataFrame: Merged DataFrame containing stock data and sentiment
                      for the test period, or None if inputs are invalid.
    """
    print(f"Aligning data for period: {start_date} to {end_date}")

    if not isinstance(stock_df.columns, pd.MultiIndex):
         print("Error: stock_df must have MultiIndex columns (e.g., (StockSymbol, Feature)).")
         return None

    if not isinstance(stock_df.index, pd.DatetimeIndex):
        try:
            stock_df.index = pd.to_datetime(stock_df.index)
        except Exception as e:
            print(f"Error converting stock_df index to DatetimeIndex: {e}")
            return None
    if not isinstance(sentiment_series.index, pd.DatetimeIndex):
        try:
            sentiment_series.index = pd.to_datetime(sentiment_series.index)
        except Exception as e:
            print(f"Error converting sentiment_series index to DatetimeIndex: {e}")
            return None

    stock_test = stock_df[(stock_df.index >= start_date) & (stock_df.index <= end_date)].copy()
    if stock_test.empty:
        print(f"Warning: No stock data found for the period {start_date} to {end_date}.")
        return None
    print(f"Filtered stock data shape: {stock_test.shape}")
    
    sentiment_test = sentiment_series[(sentiment_series.index >= start_date) & (sentiment_series.index <= end_date)].copy()
    if sentiment_test.empty:
        print(f"Warning: No sentiment data found for the period {start_date} to {end_date}.")
        sentiment_test = pd.Series(index=stock_test.index, dtype=float)

    print(f"Filtered sentiment data shape: {len(sentiment_test)}")

    merged_df = stock_test.copy() 
    merged_df['Sentiment'] = sentiment_test
    missing_sentiment_count = merged_df['Sentiment'].isnull().sum()
    total_days = len(merged_df)

    print(f"\nMerge complete. Total trading days in period: {total_days}")
    if missing_sentiment_count > 0:
        print(f"Found {missing_sentiment_count} trading days with missing sentiment scores.")
    else:
        print("No missing sentiment scores found for trading days in the period.")

    return merged_df

test_start_date = '2016-01-01'
test_end_date = '2020-01-01'

test_data_aligned = align_sentiment_with_stock_data(
    stock_features_df,
    daily_sentiment,
    test_start_date,
    test_end_date
)

if test_data_aligned is not None:
    print("\nAlignment function finished running.")
else:
    print("\nAlignment function failed.")
    
def fill_missing_sentiment(df_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the 'Sentiment' column using forward fill,
    then fills any remaining NaNs (likely at the start) with 0.

    Args:
        df_aligned (pd.DataFrame): DataFrame containing stock data and a
                                   'Sentiment' column, potentially with NaNs.

    Returns:
        pd.DataFrame: DataFrame with 'Sentiment' column NaNs filled.
                      Returns None if input is invalid or 'Sentiment' column is missing.
    """
    if 'Sentiment' not in df_aligned.columns:
        print("Error: 'Sentiment' column not found in the DataFrame.")
        return None

    print("Handling missing sentiment values...")
    initial_nan_count = df_aligned['Sentiment'].isnull().sum()
    if initial_nan_count == 0:
        print("No missing sentiment values found to fill.")
        return df_aligned

    print(f"Found {initial_nan_count} missing values. Applying forward fill (ffill)...")

    df_filled = df_aligned.copy() 
    df_filled['Sentiment'] = df_filled['Sentiment'].ffill()

    remaining_nan_count = df_filled['Sentiment'].isnull().sum()
    if remaining_nan_count > 0:
        print(f"{remaining_nan_count} NaNs remain (likely at the start). Filling with 0...")
        df_filled['Sentiment'] = df_filled['Sentiment'].fillna(0)
    else:
        print("Forward fill completed. No NaNs remaining.")

    final_nan_count = df_filled['Sentiment'].isnull().sum()
    if final_nan_count == 0:
        print("Missing sentiment values handled successfully.")
    else:
        print(f"Warning: {final_nan_count} NaNs still remain in Sentiment column after handling!")

    return df_filled

if 'test_data_aligned' in locals() and isinstance(test_data_aligned, pd.DataFrame):

    test_data_final = fill_missing_sentiment(test_data_aligned)

    if test_data_final is not None:
        print("\nFinal Test Data Info (after handling missing sentiment):")
        test_data_final.info() 
        print("\nVerifying no NaNs in 'Sentiment' column:")
        print(f"NaN count in 'Sentiment': {test_data_final['Sentiment'].isnull().sum()}")
        print("\nFirst 5 rows:")
        print(test_data_final.head())

    else:
        print("\nFailed to handle missing sentiment values.")

else:
    print("\nVariable 'test_data_aligned' not found or is not a DataFrame.")
    print("Please ensure the alignment step ran successfully and the output DataFrame")
    print("is named 'test_data_aligned' in your local environment before running this.")

def calculate_period_sentiment(df_with_daily_sentiment: pd.DataFrame, window: int = 62) -> pd.DataFrame:
    """
    Calculates the rolling average sentiment over a specified window.

    Args:
        df_with_daily_sentiment (pd.DataFrame): DataFrame containing a 'Sentiment' column
                                                with daily scores (NaNs should be handled).
        window (int): The rolling window size in days (e.g., 62 for ~2 months).

    Returns:
        pd.DataFrame: Original DataFrame with an added 'PeriodSentiment' column,
                      or None if input is invalid.
    """
    if 'Sentiment' not in df_with_daily_sentiment.columns:
        print("Error: 'Sentiment' column not found in DataFrame.")
        return None
    if df_with_daily_sentiment['Sentiment'].isnull().any():
        print("Warning: NaNs found in 'Sentiment' column. Please handle them first.")

    print(f"Calculating {window}-day rolling average sentiment...")

    df_result = df_with_daily_sentiment.copy()
    df_result['PeriodSentiment'] = df_result['Sentiment'].rolling(window=window, min_periods=1).mean() 
    df_result['PeriodSentiment'] = df_result['Sentiment'].rolling(window=window).mean()

    print("Period sentiment calculation complete.")
    final_nan_count = df_result['PeriodSentiment'].isnull().sum()
    print(f"NaN count in 'PeriodSentiment' (expected at start): {final_nan_count}")


    return df_result

if 'test_data_final' in locals() and isinstance(test_data_final, pd.DataFrame):

    sentiment_period = 62
    test_data_with_period_sentiment = calculate_period_sentiment(test_data_final, window=sentiment_period)

    if test_data_with_period_sentiment is not None:
        print("\nPeriod Sentiment added successfully:")
        print(test_data_with_period_sentiment.info())
        print("\nHead (PeriodSentiment likely NaN here):")
        print(test_data_with_period_sentiment.head(sentiment_period + 5)) 
        print("\nTail:")
        print(test_data_with_period_sentiment.tail())

    else:
        print("\nFailed to calculate period sentiment.")

else:
    print("\nVariable 'test_data_final' not found or is not a DataFrame.")
    print("Please ensure the missing sentiment values were handled and the result")
    print("is named 'test_data_final' in your local environment before running this.")

