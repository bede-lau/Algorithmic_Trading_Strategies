import numpy as np
import pandas as pd
import requests
import time
import os
from scipy import stats
import yfinance as yf
from tabulate import tabulate
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'

load_dotenv(ROOT_DIR / '.env')
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
BASE_URL = 'https://www.alphavantage.co/query'
number_of_top_stocks = 23

def get_stock_data_alpha_vantage(symbol):
    params = {
        'function' : 'OVERVIEW',
        'symbol' : symbol,
        'apikey' : API_KEY,
        'outputsize' : 'full' # 'compact' for the last 100 days of data, 'full' for all history
    }

    try:
        # Makes an API call which is considered as 1 request
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() # Raises an HTTPError for bad responses
        data = response.json()

        if 'Note' in data or 'Error Message' in data:
            print(f"Could not retrieve data for {symbol}: {data.get('Note') or data.get('Error Message')}")
            return None
        
        print(f"Successfully retrieved data for {symbol}.")
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"An error occured for {symbol}: {e}")
        return None
    
# Get total debt, cash, and latest stock price from yfinance
def get_stock_data_yf(symbol):
    stock = yf.Ticker(symbol)
    try:
        balance_sheet = stock.balance_sheet
        cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
        total_debt = balance_sheet.loc['Total Debt'].iloc[0]
        latest_price = stock.info.get('regularMarketPrice')
        return latest_price, cash, total_debt
    
    except (Exception, IndexError, KeyError) as e:
        print(f"Could not obtain latest price of {symbol}: {e}")
        return None, None, None
    
def portfolio_input():
    while True:
        try:
            portfolio_size = input("Enter the value of your portfolio: ")
            if float(portfolio_size) > 0:
                position_size = float(portfolio_size) / number_of_top_stocks
                return position_size
            else: 
                print("Please enter a positive value")
        except ValueError:
            print("Invalid value. Please enter a positive number")
            


if __name__ == "__main__":
    try:
        stocks = pd.read_csv(DATA_DIR / 'sp_500_stocks.csv')
        tickers = stocks['Ticker'].tolist()[:25]
        all_data = []

        for ticker in tickers:
            print(f"\n Obtaining data for {ticker}.")

            latest_price, cash, total_debt = get_stock_data_yf(ticker)
            if cash == None or latest_price == None or total_debt == None:
                print(f"Skipping {ticker} due to missing data from yfinance.")
                continue

            stock_data = get_stock_data_alpha_vantage(ticker)
            if stock_data == None:
                print(f"Skipping {ticker} due to missing data from Alpha Vantage.")
                continue

            pe_ratio = pd.to_numeric(stock_data.get('PERatio'), errors='coerce')
            price_to_book_ratio = pd.to_numeric(stock_data.get('PriceToBookRatio'), errors='coerce')
            price_to_sales_ratio = pd.to_numeric(stock_data.get('PriceToSalesRatioTTM'), errors='coerce')
            EV_EBITDA = pd.to_numeric(stock_data.get('EVToEBITDA'), errors='coerce')
            
            # Calculation for EV/GP
            gross_profit = pd.to_numeric(stock_data.get('GrossProfitTTM'), errors='coerce')
            shares_outstanding = pd.to_numeric(stock_data.get('SharesOutstanding'), errors='coerce')
            market_cap = pd.to_numeric(stock_data.get('MarketCapitalization'), errors='coerce')
            EV_GP = np.nan
            if gross_profit > 0 and market_cap  > 0:
                try: 
                    EV = (market_cap + total_debt) - cash
                    EV_GP = EV / gross_profit
                except (TypeError, ZeroDivisionError):
                    pass # Keep EV_GP as NaN
            

            all_data.append({
                'Ticker' : ticker, 
                'Latest Price' : latest_price,
                'Number of Shares to Buy' : '',
                'PE Ratio' : pe_ratio, 
                'Price-to-Book Ratio' : price_to_book_ratio,
                'Price-to-Sales Ratio' : price_to_sales_ratio,
                'EV/EBITDA' : EV_EBITDA,
                'EV/GP' : EV_GP,
                'RV Score' : ''
            }) 
            
            time.sleep(13)
        
        print("\n All API calls completed.")

        final_df = pd.DataFrame(all_data)
        final_df['Latest Price'] = pd.to_numeric(final_df['Latest Price'], errors='coerce')

        value_ratios = {
            'PE Ratio' : 'PE Percentile',
            'Price-to-Book Ratio' : 'PB Percentile',
            'Price-to-Sales Ratio' : 'PS Percentile',
            'EV/EBITDA' : 'EV/EBITDA Percentile',
            'EV/GP' : 'EV/GP Percentile'
        }

        # The code below loops through each ratio column once, and creates their respective percentile columns
        # Vectorization allows the percentiles from the ratios of each stock to be calculated instaneously
        for ratio, percentile_col in value_ratios.items(): #.items() converts the dictionary into a key-value pair
            final_df[percentile_col] = final_df[ratio].apply(lambda x: stats.percentileofscore(final_df[ratio].dropna(), x) if pd.notna(x) else np.nan)
            final_df[percentile_col] = final_df[percentile_col] / 100

        percentile_cols = list(value_ratios.values())
        final_df['RV Score'] = final_df[percentile_cols].mean(axis=1)
        final_df.sort_values('RV Score', ascending=True, inplace=True)
        final_df.reset_index(drop=True, inplace=True)


        position_size = portfolio_input()
        final_df['Number of Shares to Buy'] = np.floor(position_size / final_df['Latest Price'])
        print(tabulate(final_df, headers='keys', tablefmt='psql', showindex=False))

        # Convert dataframe to an Excel Table
        with pd.ExcelWriter(OUTPUT_DIR / 'High_Intrinsic_Value_Stocks.xlsx', engine = 'xlsxwriter') as writer:
            final_df.to_excel(writer, 'Value Stocks', index=False)
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
                    'num_format': '0',
                    'font_color': font_color,
                    'bg_color': background_color,
                    'border': 1
                }
            )

            float_format = writer.book.add_format(
                {
                    'num_format': '0.00',
                    'font_color': font_color,
                    'bg_color': background_color,
                    'border': 1
                }
            )

            percent_format = writer.book.add_format(
                {
                    'num_format': '0.0%',
                    'font_color': font_color,
                    'bg_color': background_color,
                    'border': 1
                }
            )

            worksheet = writer.sheets['Value Stocks']
            worksheet.set_column('A:A', 15, string_format)
            worksheet.set_column('B:B', 18, dollar_format)
            worksheet.set_column('C:C', 24, integer_format)
            worksheet.set_column('D:I', 24, float_format)
            worksheet.set_column('J:N', 18, percent_format)

    except FileNotFoundError:
        print("Error: File not found.")
    except KeyError:
        print("File must have a column named 'Ticker'.")