# GARCH Volatility Trading Strategy
# 1. Load daily price data and calculate log returns.
# 2. Fit GARCH(1,3) model in a 180-day rolling window to predict 1-day-ahead variance.
# 3. Calculate prediction premium (forecasted vs realized variance).
# 4. Generate daily trading signals when premium exceeds 1.5 standard deviations.
# 5. Backtest: forward-fill signals, calculate strategy returns.
# 6. Compare cumulative returns against buy-and-hold.

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from arch import arch_model
from pathlib import Path
import warnings
from arch.utility.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / 'results'

# 1. Load data
daily_df = pd.read_csv(DATA_DIR / 'simulated_daily_data.csv')
daily_df['Date'] = pd.to_datetime(daily_df['Date'])
daily_df = daily_df.set_index('Date')
daily_df['log_return'] = np.log(daily_df['Adj Close']).diff()

# 2. Rolling GARCH model fitting and volatility prediction
daily_df['variance'] = daily_df['log_return'].rolling(180).var()

def predict_volatility(x):
    best_model = (arch_model(y=x, p=1, q=3)
                  .fit(disp='off'))
    variance_forecast = (best_model.forecast(horizon=1)
                         .variance
                         .iloc[-1, 0])
    return variance_forecast

print('Fitting GARCH model (this takes several minutes)...')
daily_df['predictions'] = daily_df['log_return'].rolling(180).apply(lambda x: predict_volatility(x))

# 3. Calculate prediction premium
daily_df['prediction_premium'] = (daily_df['predictions'] - daily_df['variance']) / daily_df['variance']
daily_df['premium_std'] = daily_df['prediction_premium'].rolling(180).std()

# 4. Generate trading signals
daily_df['signal_daily'] = daily_df.apply(
    lambda x: 1 if (x['prediction_premium'] > x['premium_std'] * 1.5)
    else (-1 if (x['prediction_premium'] < x['premium_std'] * -1.5) else np.nan), axis=1)
daily_df['signal_daily'] = daily_df['signal_daily'].shift()

# 5. Backtest
daily_df['position'] = daily_df['signal_daily'].ffill().fillna(0)
daily_df['strategy_return'] = daily_df['position'] * daily_df['log_return']
daily_df['buy_hold_return'] = daily_df['log_return']

backtest_df = daily_df.dropna(subset=['strategy_return', 'buy_hold_return']).copy()

# 6. Performance metrics
def calc_metrics(returns, name):
    total_return = np.exp(returns.sum()) - 1
    ann_return = np.exp(returns.mean() * 252) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    cumulative = np.exp(returns.cumsum())
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    win_rate = (returns > 0).sum() / (returns != 0).sum()
    return {
        'Strategy': name,
        'Total Return': f'{total_return:.1%}',
        'Annualized Return': f'{ann_return:.1%}',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_drawdown:.1%}',
        'Win Rate': f'{win_rate:.1%}'
    }

metrics = pd.DataFrame([
    calc_metrics(backtest_df['strategy_return'], 'GARCH Strategy'),
    calc_metrics(backtest_df['buy_hold_return'], 'Buy & Hold')
])
print('\nPerformance Metrics:')
print(metrics.set_index('Strategy').to_string())

# 7. Plot cumulative returns
plt.style.use('ggplot')
cumulative = pd.DataFrame({
    'GARCH Strategy': np.exp(backtest_df['strategy_return'].cumsum()) - 1,
    'Buy & Hold': np.exp(backtest_df['buy_hold_return'].cumsum()) - 1
})
cumulative.plot(figsize=(16, 6))
plt.title('GARCH Volatility Strategy Returns Over Time')
plt.ylabel('Cumulative Return')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.savefig(RESULTS_DIR / 'garch_volatility_returns.png', dpi=150, bbox_inches='tight')
plt.show()
