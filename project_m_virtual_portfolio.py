import pandas as pd
import streamlit as st

# Function to load data from GitHub
def load_data():
    url = 'https://raw.githubusercontent.com/lmueller555/project_m/main/Project_M_Statistical_Data_Predictions4.csv'
    data = pd.read_csv(url)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def find_next_trading_day(current_date, df):
    """Find the next trading day after the current_date."""
    next_day = df[df['Date'] > current_date]['Date'].min()
    return next_day

def find_30th_trading_day(current_date, df):
    """Find the 30th trading day after the current_date."""
    future_index = df.index[df['Date'] > current_date][29]  # Get the index of the 30th day ahead
    if len(df.index[df['Date'] > current_date]) >= 30:
        return df.at[future_index, 'Date']
    else:
        return None  # In case there aren't 30 trading days available in the data


def calculate_portfolio_value(portfolio, data, current_date):
    """Calculate the total value of the portfolio for a given day."""
    portfolio_value = 0
    for ticker, holdings in portfolio.items():
        for holding in holdings:
            close_price = data[(data['Ticker'] == ticker) & (data['Date'] == current_date)]['Close/Last'].iloc[0]
            portfolio_value += holding['shares'] * close_price
    return portfolio_value

def execute_buy_logic(stock, portfolio, cash_balance, buy_percentage, data, current_date):
    ticker_to_buy = stock['ticker']
    available_cash_to_spend = cash_balance * buy_percentage
    buy_price = data[(data['Ticker'] == ticker_to_buy) & (data['Date'] == current_date)]['Close/Last'].iloc[0]
    shares_to_buy = available_cash_to_spend // buy_price
    cash_balance -= shares_to_buy * buy_price
    sell_date = find_30th_trading_day(current_date, data)
    if sell_date:
        portfolio.setdefault(ticker_to_buy, []).append({'buy_date': current_date, 'sell_date': sell_date, 'shares': shares_to_buy, 'buy_price': buy_price})
        print(f"Bought {shares_to_buy} shares of {ticker_to_buy} on {current_date} at {buy_price} per share. Scheduled to sell on {sell_date}. New cash balance: {cash_balance}.")
    else:
        print(f"Could not schedule a sale for {ticker_to_buy} bought on {current_date} due to insufficient future trading days.")
    return cash_balance


def execute_sell_logic(stock, portfolio, cash_balance, data, current_date):
    if stock['ticker'] in portfolio:
        for holding in portfolio[stock['ticker']][:]:  # Iterate over a copy since we'll modify the list
            sell_price = data[(data['Ticker'] == stock['ticker']) & (data['Date'] == current_date)]['Close/Last'].iloc[0]
            cash_balance += holding['shares'] * sell_price
            print(f"Sold {holding['shares']} shares of {stock['ticker']} on {current_date} at {sell_price} per share. New cash balance: {cash_balance}.")
            portfolio[stock['ticker']].remove(holding)  # Remove holding after selling
    return cash_balance

st.title('Project M Trading Simulation')
data = load_data()  # Load the dataset for Streamlit input and initial setup

start_date = st.date_input('Select a starting date for the simulation:', value=data['Date'].min(), min_value=data['Date'].min(), max_value=data['Date'].max())

if st.button('Run Simulation'):
    filtered_data = data[data['Date'] >= pd.to_datetime(start_date)].sort_values('Date')
    
    portfolio = {}
    cash_balance = 25000
    initial_investment = cash_balance
    buy_percentage = 0.5
    daily_portfolio_values = []
    stocks_to_buy_next_day = []  # Reintroducing this list
    stocks_to_sell_next_day = []  # Reintroducing this list

    unique_dates = filtered_data['Date'].unique()  # Iterate over each date once

    for current_date in unique_dates:
        print(f"Starting {current_date}: cash_balance = {cash_balance}")

        # Execute scheduled buys and sells for today
        for stock in stocks_to_buy_next_day[:]:  # Iterate over a copy since we'll modify the list
            if stock['buy_date'] == current_date:
                cash_balance = execute_buy_logic(stock, portfolio, cash_balance, buy_percentage, data, current_date)
                stocks_to_buy_next_day.remove(stock)

        for ticker in stocks_to_sell_next_day[:]:  # Iterate over a copy since we'll modify the list
            if ticker['sell_date'] == current_date:
                cash_balance = execute_sell_logic(ticker, portfolio, cash_balance, data, current_date)
                stocks_to_sell_next_day.remove(ticker)

        todays_data = filtered_data[filtered_data['Date'] == current_date]
        for index, row in todays_data.iterrows():
            ticker = row['Ticker']

            # Schedule buys and sells for the next day based on signals
            if row['30 Day Buy Signal'] == 1:
                next_day = find_next_trading_day(current_date, data)
                stocks_to_buy_next_day.append({'ticker': ticker, 'buy_date': next_day})

            if row['30 Day Sell Signal'] == 1:
                next_day = find_next_trading_day(current_date, data)
                stocks_to_sell_next_day.append({'ticker': ticker, 'sell_date': next_day})
                
        portfolio_value = calculate_portfolio_value(portfolio, data, current_date) + cash_balance
        daily_portfolio_values.append((current_date, portfolio_value))

    final_portfolio_value = daily_portfolio_values[-1][1]
    roi = ((final_portfolio_value - initial_investment) / initial_investment) * 100

    st.write(f"Final cash balance: {cash_balance}")
    st.write(f"Final portfolio value: {final_portfolio_value}")
    st.write(f"ROI: {roi}%")

    flattened_portfolio_data = []
    for ticker, holdings in portfolio.items():
        for holding in holdings:
            flattened_portfolio_data.append({
                'Ticker': ticker,
                'Buy Date': holding['buy_date'],
                'Sell Date': holding['sell_date'],
                'Shares': holding['shares'],
                'Buy Price': holding['buy_price']
            })

    final_portfolio_table = pd.DataFrame(flattened_portfolio_data)
    if not final_portfolio_table.empty:
        st.write("Final Portfolio:")
        st.dataframe(final_portfolio_table)
    else:
        st.write("Final Portfolio is empty.")
