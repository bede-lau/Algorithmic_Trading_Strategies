from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pandas_ta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Code Sequence:
# 1. Obtain the most recent 15 years of OHLCV and price data of all the stocks in the S&P500.
# 2. Calculate the indicator values for each unique date.
# 3. Filter the top 150 most liquid stocks for each month.
# 4. Calculate %change in monthly returns for 1M using closing prices for each date.
# 5. Import Fama-French Factors data for each month.
# 6. Use Rolling Regression OLS algorithm, with one month return as the target, to estimate the betas of each factor for each stock for the 5-year window.
# 7. Use K-Means algorithm to group 12 features (indicator values and regression coefficients) into 4 clusters for each month.
# 8. For each month, stocks in Cluster 3 are selected as the investment portfolio for that month.
# 9. For each month: calculate, optimize and assign capital allocation weightage to each stock based on Max Sharpe Ratio.
# 10. Plot portfolio returns and S&P500 returns for comparison. 


url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' 
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'}
response = requests.get(url, headers=headers)
response.raise_for_status()
sp500 = pd.read_html(response.text)[0]
sp500['Symbol'] = sp500['Symbol'].replace('.','-')
symbols_list = sp500['Symbol'].unique().tolist()

end_date = dt.datetime.today()
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*15)

df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date).stack() #.stack collapses the second header row into a column index in a MultiIndex dataframe.
df = df.drop(columns=['Adj Close'])
df.index.names = ['date', 'ticker'] # Change the index titles
df.columns.name = None # Removes 'Price'
df.columns = df.columns.str.lower() 


# Always normalize the outputs from technical indicators when their values are used in ML 

df['garman-klass_volatility'] = (((np.log(df['high']) - np.log(df['low']))**2) / 2) - (2*np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open']))**2)

def compute_rsi(close):
    rsi = pandas_ta.rsi(close=close, length=20)
    return rsi.sub(rsi.mean()).div(rsi.std())
df['rsi'] = df.groupby(level=1)['close'].transform(compute_rsi)
# .transform applies the rsi function and produces a final calculation for a selected Series, which is stored in x
# The Series is constructed by .groupby, which in this case, groups tickers of the same symbol from all dates and uses the closing prices of each date as values.
# level=1 means the Series is formed according to the values from the second index in a MultiIndex
# .transform doesn't change the size of the dataframe even with NaN values, different from .apply
# .rsi works by calculating the rsi for a minimum number of 20 closing prices or more (length=20) from the Series stored in x.

df['bb_low'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_upper'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])
# .iloc is a method to select a column or row based on integer (x,y)
# ':' in .iloc means selecting all rows or all columns depending on its position

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std())
df['atr'] = df.groupby(level=1).apply(compute_atr).droplevel(0)
# .droplevel is used to remove the an extra index column formed by the function

def compute_macd(close, fast=12, slow=26, signal=9):
    close = pd.to_numeric(close, errors='coerce').astype(float)
    if close.isna().all():
        return pd.Series(np.nan, index=close.index, dtype=float)
    
    fast_ema = close.ewm(span=fast, min_periods=fast, adjust=False).mean() 
    slow_ema = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    
    # .ewm stands for exponential weighted methods, used for calculating EMA

    mean = macd_line.mean(skipna=True)
    std = macd_line.std(skipna=True)
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=close.index, dtype=float)
    return ((macd_line - mean) / std).astype(float)

df['macd'] = df.groupby(level=1, group_keys=False)['close'].apply(compute_macd) 
df['dollar_volume'] = (df['close']*df['volume']) / 1e6


# Aggregate to monthly level and filter the top 150 most liquid stocks for each month

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
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna().sort_index(level=1, ascending=False)

# Download Fama-French Factors 
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start_date)[0]
factor_data.index = factor_data.index.to_timestamp()
factor_data.index.name = 'date'
factor_data = factor_data.resample('M').last().div(100)
factor_data.rename(columns={'HML': 'VMG'}, inplace=True)
factors = ['Mkt-RF','SMB','VMG','RMW','CMA']
regression_input = (
    data.reset_index()
    .merge(factor_data, on='date', how='inner') # understand .merge and .reset_index()
    .set_index(['date', 'ticker'])
    .sort_index()
)
regression_input['return_1m'] = regression_input['return_1m'] - regression_input['RF']
observations = regression_input.groupby('ticker').size()
valid_tickers = observations[observations >= 10].index
regression_input = (regression_input[regression_input.index.get_level_values('ticker').isin(valid_tickers)]
                    ).drop(columns=['RF','close'])

# Calculate Rolling Factor Betas (regression coefficients for each factor)
def rolling_regression(group, factors, target='return_1m', window=12*5, min_nobs=15):
    results = pd.DataFrame(
        index=group.index,
        columns=factors,
        dtype=float
    )
    depen = group[target].values # assign all values in the return_1m column to depen
    indep = group[factors].values
    non_empty_rows = ~(pd.isna(depen) | pd.isna(indep).any(axis=1)) # axis=1 means for each row in that column
    if non_empty_rows.sum() < min_nobs:
        return results
    for i in range(min_nobs-1, len(group)): # reiterate regression from the 5th observation onwards
        start_idx = max(0, i - window + 1) 
        end_idx = i + 1
        depen_data = depen[start_idx:end_idx] 
        indep_data = indep[start_idx:end_idx]
        non_empty_data = ~(pd.isna(depen_data) | pd.isna(indep_data).any(axis=1))
        if non_empty_data.sum() >= min_nobs: # rows of data should be > number of predictive factors
            depen_data = depen_data[non_empty_data] 
            indep_data = indep_data[non_empty_data] # select only rows with complete data
            try:
                reg = LinearRegression(fit_intercept=False) # fit_intercept=False means regression line goes through (0,0)
                # False when predictive factors are already standardized (common in financial time-series analysis)
                reg.fit(indep_data, depen_data) # solves the OLS equation
                results.iloc[i] = reg.coef_
            except Exception:
                continue
    return results

betas = (
    regression_input.groupby('ticker', group_keys=False) 
    # .groupby concatenates all the returned sub-dataframes (individual ticker dfs) into a single dataframe
    .apply(lambda x: rolling_regression(x, factors))
).dropna(axis=0)

feature_target_cols = [
    'garman-klass_volatility', 'rsi', 'bb_low', 'bb_mid', 'bb_upper', 'atr', 'macd',
]
betas = betas.shift().join(data[feature_target_cols], how='left').dropna(axis=0) 
# how='left' means data joins betas and discards rows that aren't matching with betas's index
betas.loc[:,factors] = betas.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
betas = betas.dropna()

# Run K-means algorithm
def get_clusters(df):
    n_clusters = 4
    if len(df) < n_clusters:
        df['cluster'] = -1
        return df
    features_cols  = df.columns.drop('cluster', errors='ignore')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features_cols])
    df['cluster'] = KMeans(n_clusters=4, 
                           random_state=0, 
    # random_stater={integer} ensures identical cluster assignments because the initial centroids are chosen the same way each time
                           init='k-means++', # Picks random points to initialize starting centroids
                           n_init=10 # Run K-means algorithms 10 seperate times to intialize centroids with the lowest inertia
                          ).fit(scaled_features).labels_
    return df

betas = betas.groupby('date', group_keys=False).apply(get_clusters)

# Point plotter for each cluster
def plot_clusters(data):
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]
    # the x- and y-values can be given in a range (must be the same number of rows)
    plt.scatter(cluster_0.iloc[:,5], cluster_0.iloc[:,3], color='red', label='Cluster 1')
    plt.scatter(cluster_1.iloc[:,5], cluster_1.iloc[:,3], color='green', label='Cluster 2')
    plt.scatter(cluster_2.iloc[:,5], cluster_2.iloc[:,3], color='blue', label='Cluster 3')
    plt.scatter(cluster_3.iloc[:,5], cluster_3.iloc[:,3], color='black', label='Cluster 4')
    plt.legend() # shows the key for the clusters (labels for each cluster)
    plt.show(block=False)
    return
plt.style.use('ggplot')
# for i in data.index.get_level_values('date').unique().to_list(): # stores every unique date in a list
#     g = data.xs(i, level='date') # for each i, g is returned a dataframe that contains rows only from that date
#     plt.title(f'Date {i}')
# plot_clusters(g) 
last_date = betas.index.get_level_values(level='date').max()
g = betas.xs(last_date, level='date')
plt.title(f'Date {last_date}')
plot_clusters(g)
    
# Based on the cluster, select stocks to invest in for each month 
filtered_df = betas[betas['cluster']==3].copy()
filtered_df = filtered_df.reset_index(level='ticker')
filtered_df.index = filtered_df.index+pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values(level='date').unique().to_list()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level='date').index.to_list()

# Download daily price data for shortlisted stocks
stocks = data.index.get_level_values('ticker').unique().to_list()
new_df = yf.download(tickers=stocks, 
                      start=data.index.get_level_values('date').min()-pd.DateOffset(months=12),
                      end=data.index.get_level_values('date').max())
# [0] and [-1] picks the first and last value in the list

# Portfolio optimization function using Max Efficient Frontier
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

# Visualize portfolio returns and compare to SPX500 returns
end_date = dt.datetime.today()
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*15)
spy = yf.download(tickers='SPY', start=start_date, end=end_date)
spy_return = np.log(spy['Close']).diff().dropna().rename({'SPY':'SPY Buy&Hold'}, axis=1)
spy_return.columns.name = None
portfolio_df = portfolio_df.merge(spy_return, left_index=True, right_index=True).sort_index() # merge portfolio and SPY daily returns
plt.style.use('ggplot')
portfolio_cumulative_returns = np.exp(np.log1p(portfolio_df).cumsum()) # log1p means log(1+x) but handles floating-piont precision better
# .cumsum() calculates the total return relative to the start and .exp converts it to a growth factor
# Totalling logged returns will give the logged growth factor (e.g., ln(1.3))
# To obtain the growth factor, .exp will cancel out the ln
total_percentage_return = portfolio_cumulative_returns - 1
total_percentage_return[:end_date].plot(figsize=(16,6))
plt.title('Unsupervised Machine Learning Strategy Returns Over Time')
plt.ylabel('Returns')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.show(block=False)
plt.show()