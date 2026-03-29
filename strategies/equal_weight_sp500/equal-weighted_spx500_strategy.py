import yfinance as yf
import xlsxwriter
from tabulate import tabulate
import pandas as pd
import math
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'output'

# Appends all stocks' ticker into a list
try:
    stocks = pd.read_csv(DATA_DIR / 'sp_500_stocks.csv')
    ticker_list = stocks['Ticker'].tolist()[:300]
except FileNotFoundError:
    print("CSV file not found.")
    ticker_list = []

list_of_rows = []

# Obtain the symbols, prices and market caps, and appends it to a list of dictionaries
if ticker_list:
    print(f"Extracting {len(ticker_list)} stock data...")
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker) # Create a ticker object
            price = stock.info.get('regularMarketPrice')
            market_cap = stock.info.get('marketCap')
            final_market_cap = math.floor(market_cap) if market_cap is not None else None
            list_of_rows.append({'Symbol': ticker, 'Stock Price': price, 'Market Capitalization' : final_market_cap})

            # 0.5 second delay to prevent getting banned
            time.sleep(0.5)
        except Exception:
            print(f"Could not obtain data for {ticker}.") 

# Construct Dataframe
final_dataframe = pd.DataFrame(list_of_rows, columns = ['Symbol', 'Stock Price', 'Market Capitalization', 'Number of Shares To Buy'])
final_dataframe.dropna(inplace=True, subset=['Stock Price'])

# Number of shares to buy calculation
portfolio_size = input("Enter the value of your portfolio: ")

try:
    equity_value = float(portfolio_size)
    position_size = equity_value / len(final_dataframe)
    final_dataframe['Number of Shares To Buy'] = (position_size / final_dataframe['Stock Price']).astype(float)

except (ValueError, ZeroDivisionError):
    print("That's not a valid number. Please try again.")

print(tabulate(final_dataframe, headers='keys', tablefmt='psql', showindex=False))

# Convert the dataframe into an Excel Sheet
writer = pd.ExcelWriter(OUTPUT_DIR / 'Recommended Trades.xlsx', engine = 'xlsxwriter')
final_dataframe.to_excel(writer, 'Recommended Trades', index=False)
background_color = '#0a0a23'
font_color = '#ffffff' 

string_format = writer.book.add_format(
    {
        'font_color': font_color,
        'bg_color': background_color,
        'border': 1
    }
)

dollar_format = writer.book.add_format(
    {
        'num_format': '$0.00',
        'font_color': font_color,
        'bg_color': background_color,
        'border': 1
    }
)

integer_format = writer.book.add_format(
    {
        'num_format': '0.00',
        'font_color': font_color,
        'bg_color': background_color,
        'border': 1
    }
)
writer.sheets['Recommended Trades'].set_column('A:A', 18, string_format)
writer.sheets['Recommended Trades'].set_column('B:C', 24, dollar_format)
writer.sheets['Recommended Trades'].set_column('D:D', 24, integer_format)
writer.close()