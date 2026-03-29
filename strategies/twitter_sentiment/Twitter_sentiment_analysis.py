# 1. Load Twitter sentiment data on NASDAQ stocks.
# 2. Set date and symbol as the index.
# 3. Calculate engagement ratio.
# 4. Rank the top 5 stocks with the highest engagement ratio, number of likes, and number of comments.
# 5. Create the portfolio at the start of each new month.
# 6. Download NASDAQ price data and compare returns to portfolio returns.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf
from pathlib import Path
plt.style.use('ggplot')

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
sentiment_df = pd.read_csv(DATA_DIR / 'sentiment_data.csv')
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df.set_index(['date', 'symbol']) # convert dataframe to MultiIndex
sentiment_df['engagement_ratio'] = sentiment_df['twitterComments'] / sentiment_df['twitterLikes']
sentiment_df = sentiment_df[ # ensure the stock is popular and the engagement is genuine
    (sentiment_df['engagement_ratio'] > 0.1) & 
    (sentiment_df['twitterLikes'] > 20) & 
    (sentiment_df['twitterComments'] > 20)]

aggregated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='ME'), 'symbol']))[['engagement_ratio']].mean()
# [pd.Grouper(freq='M'), 'symbol'] means select every unique symbol for every calendar month
# [[]] double square brackets keeps the result as a dataframe instead of collapsing into a Series. Typically used during .groupby()/selection aggregations.
aggregated_df['rank'] = aggregated_df.groupby(level=0)['engagement_ratio'].rank(ascending=False)
filtered_df = aggregated_df[aggregated_df['rank'] < 6]
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])

dates = filtered_df.index.get_level_values(level=0).unique().tolist()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.to_list()

stocks_list = sentiment_df.index.get_level_values(level='symbol').unique().to_list()
first_date = pd.Timestamp(dates[0]) - pd.DateOffset(years=1)
last_date = pd.Timestamp(dates[-1]) + pd.DateOffset(years=1)
prices = yf.download(tickers=stocks_list, start=first_date, end=last_date)['Close']

returns_df = np.log(prices).diff().stack().dropna().unstack() # calculate the difference in logged prices to obtain returns
portfolio_df = pd.DataFrame()
for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d') # gives the end date of the same month
    cols = fixed_dates[start_date]
    available_cols = [col for col in cols if col in returns_df.columns] 
    # only aggregate symbols in fixed_dates for that date if they're in returns_df
    # some symbols may be missing after .diff() calculation
    temp_df = returns_df.loc[start_date:end_date, available_cols].mean(axis=1).to_frame('Portfolio Return')
    # calculate the average return across the top 5 stocks for each date
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

# Downlaod Nasdaq OHLCV data
qqq_df = yf.download(tickers='QQQ', start=first_date, end=last_date)
qqq_df.columns = qqq_df.columns.droplevel(1)
qqq_return = np.log(qqq_df['Close']).diff().to_frame('Nasdaq Return')
portfolio_df = portfolio_df.merge(qqq_return, left_index=True, right_index=True)
portfolio_cumulative_return = np.exp(portfolio_df.cumsum()).sub(1)
portfolio_cumulative_return.plot(figsize=(16,6))
plt.title('Twitter Engagement Ratio Strategy Return Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
RESULTS_DIR = DATA_DIR.parent / 'results'
plt.savefig(RESULTS_DIR / 'twitter_sentiment_returns.png', dpi=150, bbox_inches='tight')
plt.show()

