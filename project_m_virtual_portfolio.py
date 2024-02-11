import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Function to load data from GitHub
def load_data():
    url = 'https://raw.githubusercontent.com/lmueller555/project_m/main/Project_M_Statistical_Data_Predictions4.csv'
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to calculate total portfolio value
def calculate_total_portfolio_value(portfolio, current_cash, todays_data):
    total_value = current_cash
    for ticker, details in portfolio.items():
        current_price = todays_data[todays_data['Ticker'] == ticker].iloc[-1]['Close/Last'] if not todays_data[todays_data['Ticker'] == ticker].empty else 0
        total_value += details['quantity'] * current_price
    return total_value

# Function to create a DataFrame from the portfolio for display and calculate total portfolio value
def create_portfolio_display_df_and_value(portfolio, data, selected_date):
    holdings = []
    total_value = 0
    for ticker, details in portfolio.items():
        latest_data = data[data['Ticker'] == ticker].iloc[-1]  # Get the last available data
        current_price = latest_data['Close/Last']
        current_value = details['quantity'] * current_price
        total_value += current_value  # Add to total portfolio value
        holdings.append({
            'Ticker': ticker,
            'Buy Date': details['buy_date'].strftime('%Y-%m-%d'),
            'Quantity': details['quantity'],
            'Buy Price': details['buy_price'],
            'Current Price': current_price,
            'Current Value': current_value  # Add the current value
        })
    return pd.DataFrame(holdings), total_value

def calculate_total_portfolio_value(portfolio, current_cash, todays_data, last_known_prices):
    total_value = current_cash
    for ticker, details in portfolio.items():
        # Check if today's data has the ticker's info, otherwise use the last known price
        if not todays_data[todays_data['Ticker'] == ticker].empty:
            current_price = todays_data[todays_data['Ticker'] == ticker]['Close/Last'].iloc[-1]
            last_known_prices[ticker] = current_price  # Update the last known price
        else:
            # If the ticker is not in today's data, use the last known price
            current_price = last_known_prices.get(ticker, details['buy_price'])  # Default to buy price if no price is known

        total_value += details['quantity'] * current_price
    return total_value, last_known_prices

def run_simulation(data, start_date):
    start_date = pd.Timestamp(start_date)
    portfolio = {}  # Reset the portfolio at the start of the simulation
    current_cash = 25000  # Reset the cash balance at the start of the simulation
    portfolio_value_history = []

    # Get the trading dates from the dataset starting from the selected start date
    trading_dates = data[data['Date'] >= start_date]['Date'].sort_values().unique()

    # Iterate through each trading date
    for trading_index, single_date in enumerate(trading_dates):
        todays_data = data[data['Date'] == single_date]
        # Create a copy of the current portfolio for iterating
        current_portfolio = portfolio.copy()
        for ticker, details in current_portfolio.items():
            buy_date_index = np.where(trading_dates == details['buy_date'])[0][0]
            # Check if 30 trading days have passed
            if trading_index - buy_date_index >= 30:
                # Fetch sell data for this date
                sell_data = todays_data[todays_data['Ticker'] == ticker]
                if not sell_data.empty:
                    sell_price = sell_data.iloc[0]['Open']
                    earnings = details['quantity'] * sell_price
                    current_cash += earnings
                    print(f"Selling {details['quantity']} shares of {ticker} at {sell_price}, earning {earnings}")
                    del portfolio[ticker]  # Remove stock from portfolio after selling

        # Implementing the buy logic
        for _, row in todays_data.iterrows():
            if row['30 Day Buy Signal'] == 1 and current_cash > 0:
                investment_amount = current_cash * 0.50
                quantity = int(investment_amount // row['Open'])
                if quantity > 0:
                    current_cash -= quantity * row['Open']
                    portfolio[row['Ticker']] = {
                        'buy_date': single_date,
                        'quantity': quantity,
                        'buy_price': row['Open']
                    }
                    print(f"Buying {quantity} shares of {row['Ticker']} at {row['Open']}, spending {quantity * row['Open']}")

        # Calculate and print the portfolio value after each day
        total_portfolio_value = calculate_total_portfolio_value(portfolio, current_cash, todays_data)
        portfolio_value_history.append({'Date': single_date, 'Portfolio Value': total_portfolio_value})
        print(f"Date: {single_date}, Cash: {current_cash}, Portfolio Value: {total_portfolio_value}")

    # Update session state after simulation
    st.session_state.portfolio = portfolio
    st.session_state.current_cash = current_cash
    st.session_state.portfolio_value_history = portfolio_value_history


# Streamlit app UI setup
st.title("Paper Trading Simulator")
data = load_data()

# Initialize session state variables
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'current_cash' not in st.session_state:
    st.session_state.current_cash = 25000
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = data['Date'].min()

# Date picker for selecting simulation start date
selected_start_date = st.date_input("Select a start date for simulation", value=data['Date'].min())

# Button to run simulation
if st.button('Run Simulation from Selected Day'):
    run_simulation(data, selected_start_date)

    # Display final portfolio and cash balance
    st.write("### Final Portfolio")
    portfolio_df, total_portfolio_value = create_portfolio_display_df_and_value(
        st.session_state.portfolio, data, data['Date'].max()
    )
    st.table(portfolio_df)

    st.write(f"### Final Cash Balance: ${st.session_state.current_cash:,.2f}")
    st.write(f"### Total Portfolio Value: ${total_portfolio_value:,.2f}")

    # Display historical portfolio value
    portfolio_value_history_df = pd.DataFrame(st.session_state.portfolio_value_history)
    portfolio_value_history_df['Date'] = pd.to_datetime(portfolio_value_history_df['Date'])
    st.line_chart(portfolio_value_history_df.set_index('Date')['Portfolio Value'])

# Ensure the date_input widget reflects the current state
st.session_state.selected_date = selected_start_date
