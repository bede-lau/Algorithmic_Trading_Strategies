from sklearn.cluster import KMeans
import matplotlib.ticker as mtick
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import requests
import warnings
warnings.filterwarnings('ignore')

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' 
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'}
response = requests.get(url, headers=headers)
response.raise_for_status()
sp500 = pd.read_html(response.text)[0]
sp500['Symbol'] = sp500['Symbol'].replace('.', '-')
symbolslist = sp500['Symbol'].unique().tolist() 

end_date = dt.datetime.today()
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*15)

df = yf.download(tickers=symbolslist,
                 start=start_date,
                 end=end_date).stack()
df = df.drop(columns=['Adj Close'])
df.index.names = ['date', 'ticker']
df.columns.name = None
df.columns = df.columns.str.lower()


df['garman-klass_volatility'] = (((np.log(df['high']) - np.log(df['low']))**2) / 2) - (2*np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open']))**2)
# RSI Indicator
def compute_rsi(close):
    rsi = pandas_ta.rsi(close=close, length=20)
    return rsi.sub(rsi.mean()).div(rsi.std())
df['rsi'] = df.groupby(level=1)['close'].transform(compute_rsi)
# .transform applies the rsi function and produces a final calculation for a selected Series, which is stored in x
# The Series is constructed by .groupby, which in this case, groups tickers of the same symbol from all dates and uses the closing prices of each date as values.
# level=1 means the Series is formed according to the values from the second index in a MultiIndex
# .transform doesn't change the size of the dataframe even with NaN values, different from .apply
# .rsi works by calculating the rsi for a minimum number of 20 closing prices or more (length=20) from the Series stored in x.

# Bollinger Bands Indicator
df['bb_low'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_upper'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])
# .iloc is a method to select a column or row based on integer (x,y)
# ':' in .iloc means selecting all rows or all columns depending on its position

# ATR Indicator
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std())
df['atr'] = df.groupby(level=1).apply(compute_atr).droplevel(0)
# .droplevel is used to remove the an extra index column formed by the function when 

def compute_macd(close, fast=12, slow=26):
    close = pd.to_numeric(close, errors='coerce').astype(float)
    if close.isna().all():
        return pd.Series(np.nan, index=close.index, dtype=float)
    
    fast_ema = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema

    mean = macd_line.mean(skipna=True)
    std = macd_line.std(skipna=True)
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=close.index, dtype=float)
    return ((macd_line - mean) / std).astype(float)

df['macd'] = df.groupby(level=1, group_keys=False)['close'].apply(compute_macd) 
# lambda x: is only necessary when the function being applied needs an extra argument

df['dollar_volume'] = (df['close']*df['volume']) / 1e6


last_cols = [c for c in df.columns.unique(level=0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
agg_data = pd.concat([df['dollar_volume'].groupby(level=1).resample('M', level=0).mean().to_frame('dollar_volume'), 
          df[['close'] + last_cols].groupby(level=1).resample('M', level=0).last()], axis=1).dropna()
# pd.concat() allows two or more Series or dataframes in a list to be concatenated horizontally or vertically
# axis=1 means concatenate by columns   
# df['dollar_volume'] selects the dollar_volume column
# .groupby() selects the other index in a MultiIndex dataframe
# .resample() selects the Datetimeindex
# .to_frame converts the returned Series (because only one column was selected) to a dataframe

liquidity = df['dollar_volume'].unstack('ticker').rolling(5*12).mean().stack().to_frame('dollar_volume')
liquidity['dollar_vol_rank'] = liquidity.groupby(level=0)['dollar_volume'].rank(ascending=False)
data = agg_data.join(liquidity['dollar_vol_rank']) # .join() is used to combine columns or combine dataframes horizontally (similar to .concatenate())
data = data[data['dollar_vol_rank'] < 151].dropna(axis=0)
data = data.drop(columns=['dollar_vol_rank','dollar_volume'])

# Calculate the %change in monthly returns for different time horizons to capture the correlations between high returns and indicator values
def calculate_returns(df):
    outlier_cutoff = 0.005 # Returns above and below the 99.5 percentile of that stock are not factored in
    lags = [1]
    for lag in lags:
        df[f'return_{lag}m'] = (df['close']
                            .pct_change(lag)
                            .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1-outlier_cutoff)))
                            .add(1).pow(1/lag).sub(1) # Calculate the monthly compounded percentage return
                            )
    return df
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna().sort_index(level=1)


def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4, 
                           random_state=0, 
    # random_stater={integer} ensures identical cluster assignments because the initial centroids are chosen the same way each time
                           init='random', # Picks random points to initialize starting centroids
                           n_init=10
                          ).fit(df).labels_
    return df
data = data.dropna().drop(columns='close').groupby('date', group_keys=False).apply(get_clusters)

def plot_clusters(data):
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]
    plt.scatter(cluster_0.iloc[:,5], cluster_0.iloc[:,3], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,5], cluster_1.iloc[:,3], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,5], cluster_2.iloc[:,3], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,5], cluster_3.iloc[:,3], color='black', label='cluster 3')
    plt.legend() # shows the key for the clusters (labels for each cluster)
    plt.show(block=False)
    return
plt.style.use('ggplot')
for i in data.index.get_level_values('date').unique().to_list(): # stores every unique date in a list
    g = data.xs(i, level=1) # for each i, g is returned a dataframe that contains rows only from that date
    # plt.title(f'Date {i}')
plot_clusters(g)
    
filtered_df = data[data['cluster']==3].copy()
filtered_df = filtered_df.reset_index(level='ticker')
filtered_df.index = filtered_df.index+pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values(level='date').unique().to_list()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level='date').index.to_list()

stocks = data.index.get_level_values('ticker').unique().to_list()
new_df = yf.download(tickers=stocks, 
                      start=data.index.get_level_values('date').min()-pd.DateOffset(months=12),
                      end=data.index.get_level_values('date').max())


def optimize_weights(prices, lower_bound=0): # determine the weightage assigned to each stock
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252) # returns the annualized expected returns of each stock
    cs = risk_models.CovarianceShrinkage(prices, frequency=252)
    cov = cs.ledoit_wolf() # returns the annualized sample covariance matrix of asset returns
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, 1),
                           solver='SCS')
    ef.max_sharpe()
    return ef.clean_weights() 

returns_dataframe = np.log(new_df['Close']).diff() # returns a MultiIndex Series of logged prices subtracted from the previous date

# Optimize portfolio for each month
portfolio_dfs = []
for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d') # 0 in MonthEnd() means stay in the current month
    cols = fixed_dates[start_date]
    optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d') # strftime formats the date as a string
    optimization_df = new_df.loc[optimization_start_date:optimization_end_date,'Close'][cols]

    # Assign weights to each stock according to Max Sharpe Ratio. If failure occurs, assign equal weights to stocks
    success = False
    try:
        weights = optimize_weights(optimization_df, lower_bound=0.001)
        weights = pd.DataFrame(weights, index=[0])
        success = True
    except ValueError:
        print(f"Max Sharpe Ratio calculation failed for {start_date}, assigning equal-weights to all stocks")
    if success == False:
        weights = pd.DataFrame(
            [1/len(optimization_df.columns) for i in range(len(optimization_df.columns))], 
            index=optimization_df.columns.to_list(),
            columns=pd.Series(0)).T
        
    temp_df = returns_dataframe[start_date:end_date] # obtain returns for the current month
    # reset the date level for the "returns" and "weights" Series, then merge the two
    temp_df = temp_df.stack().to_frame('Return').reset_index(level=0).merge(
        weights.stack().to_frame('Weight').reset_index(level=0, drop=True), # drop=True discards the index level that's been converted into a column
        left_index=True, right_index=True # only include rows where the index values match in both dataframes
    ).reset_index().set_index(['Date', 'Ticker'])
    temp_df['Weighted Return'] = temp_df['Return'] * temp_df['Weight']
    
    # returns a Series of sum of portfolio return for each month
    temp_df = temp_df.groupby(level=0)['Weighted Return'].sum().to_frame('Portfolio Return') 
    portfolio_dfs.append(temp_df) # append dataframe to list
portfolio_df = pd.concat(portfolio_dfs).drop_duplicates()
portfolio_df.plot()

end_date = dt.datetime.today()
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*15)
spy = yf.download(tickers='SPY', start=start_date, end=end_date)
spy_return = np.log(spy['Close']).diff().dropna().rename({'SPY':'SPY Buy&Hold'}, axis=1)
spy_return.columns.name = None
portfolio_df = portfolio_df.merge(spy_return, left_index=True, right_index=True).sort_index()
plt.style.use('ggplot')
portfolio_cumulative_returns = np.exp(np.log1p(portfolio_df).cumsum()) # log1p means log(1+x) but handles floating-piont precision better
# .cumsum() calculates the total return relative to the start and .exp converts it to a growth factor
# Totalling logged returns will give the logged growth factor (e.g., ln(1.3))
# To obtain the growth factor, .exp will cancel out the ln
total_portfolio_returns = portfolio_cumulative_returns - 1
total_portfolio_returns[:end_date].plot(figsize=(16,6))
plt.title('Unsupervised Machine Learning Strategy Returns Over Time')
plt.ylabel('Returns')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.show(block=False)
plt.show()
