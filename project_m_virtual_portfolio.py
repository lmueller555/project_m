import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Function to load data from GitHub


def load_data():
    url = 'https://raw.githubusercontent.com/lmueller555/project_m/main/Project_M_Statistical_Data_Predictions4.csv'
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to simulate trading for a single day


def simulate_trading(data, date):
    portfolio = st.session_state.portfolio
    current_cash = st.session_state.current_cash

    # Sell logic: Check for stocks held for 30 days and sell them
    for ticker, details in list(portfolio.items()):
        if (date - details['buy_date']).days >= 30:
            sell_data = data[(data['Ticker'] == ticker) & (
                data['Date'] == details['buy_date'] + timedelta(days=30))]
            if not sell_data.empty:
                sell_price = sell_data.iloc[0]['Open']
                earnings = details['quantity'] * sell_price
                current_cash += earnings
                # Remove stock from portfolio after selling
                del portfolio[ticker]

    # Buy logic: Check for buy signals and execute trades
    todays_data = data[data['Date'] == date]
    for _, row in todays_data.iterrows():
        if row['30 Day Buy Signal'] == 1 and current_cash > 0:
            investment_amount = current_cash * 0.50
            quantity = int(investment_amount // row['Open'])
            if quantity > 0:
                current_cash -= quantity * row['Open']
                portfolio[row['Ticker']] = {
                    'buy_date': date,
                    'quantity': quantity,
                    'buy_price': row['Open']
                }

    # Update session state
    st.session_state.portfolio = portfolio
    st.session_state.current_cash = current_cash

# Function to create a DataFrame from the portfolio for display and calculate total portfolio value


def create_portfolio_display_df_and_value(portfolio, data, selected_date):
    holdings = []
    # Start with the current cash balance
    total_value = st.session_state.current_cash
    for ticker, details in portfolio.items():
        latest_data = data[(data['Ticker'] == ticker) &
                           (data['Date'] <= selected_date)]
        if not latest_data.empty:
            current_price = latest_data.iloc[-1]['Close/Last']
            market_value = details['quantity'] * current_price
            # Add the market value of this stock to the total portfolio value
            total_value += market_value
        else:
            current_price = 'N/A'
            market_value = 'N/A'

        holdings.append({
            'Ticker': ticker,
            'Buy Date': details['buy_date'].strftime('%Y-%m-%d'),
            'Quantity': details['quantity'],
            'Buy Price': details['buy_price'],
            'Current Price': current_price,
            'Market Value': market_value
        })
    return pd.DataFrame(holdings), total_value


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
if 'portfolio_value_history' not in st.session_state:
    st.session_state.portfolio_value_history = []

# Date picker for selecting simulation date
selected_date = st.date_input("Select a date for simulation",
                              value=st.session_state.selected_date.date(),
                              min_value=data['Date'].min().date(),
                              max_value=data['Date'].max().date())

# Button to execute trades
if st.button('Execute Trades for Selected Day'):
    simulate_trading(data, pd.Timestamp(selected_date))
    # After trades are executed, update portfolio display and total value
    portfolio_df, total_portfolio_value = create_portfolio_display_df_and_value(
        st.session_state.portfolio, data, pd.Timestamp(selected_date))
    st.session_state.selected_date = pd.Timestamp(
        selected_date)  # Update selected date
    st.session_state.portfolio_value_history.append(
        {'Date': selected_date, 'Portfolio Value': total_portfolio_value})
    st.experimental_rerun()

# Display the portfolio table and total portfolio value
st.write("### Current Portfolio")
portfolio_df, total_portfolio_value = create_portfolio_display_df_and_value(
    st.session_state.portfolio, data, st.session_state.selected_date
)
st.table(portfolio_df)
st.write(f"### Total Portfolio Value: ${total_portfolio_value:,.2f}")

# Update and display the cash balance
st.write(f"### Current Cash Balance: ${st.session_state.current_cash:,.2f}")

# Visualization of historical portfolio value
st.write("### Historical Portfolio Value")

# Ensure portfolio_value_history is initialized correctly
if 'portfolio_value_history' not in st.session_state or isinstance(st.session_state.portfolio_value_history, list):
    st.session_state.portfolio_value_history = pd.DataFrame(
        columns=['Date', 'Portfolio Value'])
elif not isinstance(st.session_state.portfolio_value_history, pd.DataFrame):
    # Convert existing data into a DataFrame
    st.session_state.portfolio_value_history = pd.DataFrame(
        st.session_state.portfolio_value_history)

# Check if the date is already present; if not, append the new value
if pd.Timestamp(st.session_state.selected_date) not in pd.to_datetime(st.session_state.portfolio_value_history['Date']).values:
    new_record = pd.DataFrame({
        'Date': [pd.Timestamp(st.session_state.selected_date)],
        'Portfolio Value': [total_portfolio_value]
    })
    st.session_state.portfolio_value_history = st.session_state.portfolio_value_history.append(
        new_record, ignore_index=True)

# Convert 'Date' to datetime format if it's not already, and set it as index for plotting
st.session_state.portfolio_value_history['Date'] = pd.to_datetime(
    st.session_state.portfolio_value_history['Date'])
portfolio_value_history_for_plot = st.session_state.portfolio_value_history.set_index(
    'Date')

st.line_chart(portfolio_value_history_for_plot['Portfolio Value'])
