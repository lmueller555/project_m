import pandas as pd
import streamlit as st

# Streamlit UI for selecting start date
st.title('Project M Trading Simulation')
selected_start_date = st.date_input("Select a Start Date for the Simulation")
# Convert selected start date to datetime format to match dataset
start_date = pd.to_datetime(selected_start_date)

# Button to start the simulation
if st.button('Start Simulation'):

    # Load the dataset directly from GitHub
    dataset_url = 'https://raw.githubusercontent.com/lmueller555/project_m/main/Updated_Dataset_with_Indicators_Test2.csv'
    data_sorted = pd.read_csv(dataset_url)
    data_sorted['Date'] = pd.to_datetime(data_sorted['Date'])
    data_sorted = data_sorted.sort_values(by='Date')

    # Initialize variables for the simulation
    unique_dates = sorted(data_sorted['Date'].unique())
    initial_cash = 25000  # Adjust as necessary
    cash = initial_cash
    portfolio = {}  # To track stocks bought
    sell_signals = {}  # To track sell signals for next trading day
    pending_buy_orders = {}  # To track pending buy orders until execution
    portfolio_value_records = []  # Use a list to collect records

    # Create a date index mapping
    date_index = {date: idx for idx, date in enumerate(unique_dates)}

    # Adjusted to start from the selected start date
    if start_date in unique_dates:
        start_index = unique_dates.index(start_date)
    else:
        st.write("Selected start date is not available in the dataset.")
        st.stop()

    # Simulation starts here
    for current_date in unique_dates[start_index:-1]:
        today_index = date_index[current_date]
        today_data = data_sorted[data_sorted['Date'] == current_date]

        # Execute pending buy orders for today
        if current_date in pending_buy_orders:
            for order in pending_buy_orders[current_date]:
                # Cash deduction and portfolio update for each pending order
                cash -= order['amount_to_spend']
                shares_to_buy = order['amount_to_spend'] / \
                    order['next_day_open_price']
                if order['sell_date'] not in portfolio:
                    portfolio[order['sell_date']] = {}
                portfolio[order['sell_date']][order['company']] = {
                    'shares': shares_to_buy, 'buy_price': order['next_day_open_price']}
                print(f"Executed Buy Order for {order['company']} on {current_date.strftime('%Y-%m-%d')} at ${order['next_day_open_price']:.2f} for {shares_to_buy:.2f} shares. Amount Spent: ${order['amount_to_spend']:.2f}. Will sell on: {order['sell_date'].strftime('%Y-%m-%d')}")

            del pending_buy_orders[current_date]

        # The rest of the code for handling sell signals, updating portfolio for the day
        # Check for 30 Day Sell Signals and queue them for next trading day
        for index, row in today_data.iterrows():
            if row['30 Day Sell Signal'] == 1:
                next_sell_day = unique_dates[min(
                    today_index + 1, len(unique_dates)-1)]
                if next_sell_day not in sell_signals:
                    sell_signals[next_sell_day] = []
                sell_signals[next_sell_day].append(row['Company'])

        # Buy condition: Check for buy signals and place orders for 14 days later
        for index, row in today_data.iterrows():
            if row['30 Day Buy Signal'] == 1 and today_index + 14 < len(unique_dates):
                execution_day = unique_dates[today_index + 14]
                if execution_day not in pending_buy_orders:
                    pending_buy_orders[execution_day] = []

                next_day_data = data_sorted[(data_sorted['Date'] == execution_day) & (
                    data_sorted['Company'] == row['Company'])]
                if not next_day_data.empty:
                    next_day_open_price = next_day_data['Open'].values[0]
                    amount_to_spend = cash * 0.5
                    pending_buy_orders[execution_day].append({
                        'company': row['Company'],
                        'amount_to_spend': amount_to_spend,
                        'next_day_open_price': next_day_open_price,
                        'sell_date': unique_dates[min(today_index + 22, len(unique_dates)-1)]
                    })

        # Executing sell orders from sell signals identified on the previous day
        if current_date in sell_signals:
            for stock in sell_signals[current_date]:
                if stock in portfolio:
                    for sell_date, info in portfolio.items():
                        if stock in info:
                            sell_price = data_sorted[(data_sorted['Date'] == current_date) & (
                                data_sorted['Company'] == stock)]['Open'].values[0]
                            cash += sell_price * info[stock]['shares']
                            print(
                                f"Sell Signal Executed: Sold {stock} for ${sell_price:.2f}/share, total ${sell_price * info[stock]['shares']:.2f}. New cash balance: ${cash:.2f}")
                            del portfolio[sell_date][stock]
            del sell_signals[current_date]

        # Update portfolio value for current day
        current_portfolio_value = cash  # Include current cash in portfolio value
        for _, stocks in portfolio.items():
            for stock, info in stocks.items():
                stock_data_today = data_sorted[(data_sorted['Date'] == current_date) & (
                    data_sorted['Company'] == stock)]
                if not stock_data_today.empty:
                    last_price = stock_data_today['Open'].iloc[0]
                    current_portfolio_value += last_price * info['shares']
        portfolio_value_records.append(
            {'Date': current_date, 'Value': current_portfolio_value})

    # Convert the records list to a DataFrame for plotting
    portfolio_values_df = pd.DataFrame(portfolio_value_records)
    # Plot the portfolio value over time
    st.line_chart(portfolio_values_df.set_index('Date'))

    # Final calculations and displaying results...
    final_portfolio_value = portfolio_values_df['Value'].iloc[-1] if not portfolio_values_df.empty else initial_cash
    roi = ((final_portfolio_value - initial_cash) / initial_cash) * 100
    st.write("Simulation has ended.")
    st.write(f"Initial Investment: ${initial_cash:.2f}")
    st.write(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    st.write(f"Current Cash: ${cash:.2f}")
    st.write(f"ROI: {roi:.2f}%")




# import pandas as pd
# import streamlit as st

# # Streamlit UI for selecting start date
# st.title('Project M Trading Simulation')
# selected_start_date = st.date_input("Select a Start Date for the Simulation")
# # Convert selected start date to datetime format to match dataset
# start_date = pd.to_datetime(selected_start_date)

# # Button to start the simulation
# if st.button('Start Simulation'):

#     # Load the dataset directly from GitHub
#     dataset_url = 'https://raw.githubusercontent.com/lmueller555/project_m/main/Updated_Dataset_with_Indicators_Test2.csv'
#     data_sorted = pd.read_csv(dataset_url)
#     data_sorted['Date'] = pd.to_datetime(data_sorted['Date'])
#     data_sorted = data_sorted.sort_values(by='Date')

#     # Initialize variables for the simulation
#     unique_dates = sorted(data_sorted['Date'].unique())
#     initial_cash = 25000  # Adjust as necessary
#     cash = initial_cash
#     portfolio = {}  # To track stocks bought
#     sell_signals = {}  # To track sell signals for next trading day
#     portfolio_value_records = []  # Use a list to collect records

#     # Create a date index mapping
#     date_index = {date: idx for idx, date in enumerate(unique_dates)}

#     # Adjusted to start from the selected start date
#     if start_date in unique_dates:
#         start_index = unique_dates.index(start_date)
#     else:
#         st.write("Selected start date is not available in the dataset.")
#         st.stop()

#     # Simulation starts here
#     for current_date in unique_dates[start_index:-1]:
#         today_index = date_index[current_date]
#         today_data = data_sorted[data_sorted['Date'] == current_date]

#         # Print current trading day
#         # Use Streamlit's write function for output in the app
#         print(f"Trading Day: {current_date.strftime('%Y-%m-%d')}")

#         # Execute sell orders from sell signals identified on the previous day
#         if current_date in sell_signals:
#             for stock in sell_signals[current_date]:
#                 if stock in portfolio:
#                     for sell_date, info in portfolio.items():
#                         if stock in info:
#                             sell_price = data_sorted[(data_sorted['Date'] == current_date) & (
#                                 data_sorted['Company'] == stock)]['Open'].values[0]
#                             cash += sell_price * info[stock]['shares']
#                             print(
#                                 f"Sell Signal Executed: Sold {stock} for ${sell_price:.2f}/share, total ${sell_price * info[stock]['shares']:.2f}. New cash balance: ${cash:.2f}")
#                             # Remove sold stock from portfolio
#                             del portfolio[sell_date][stock]
#             # Clear sell signals for executed day
#             del sell_signals[current_date]

#         # Sell condition: Check if any stocks are to be sold today (existing condition)
#         if current_date in portfolio:
#             for stock, info in portfolio[current_date].items():
#                 sell_price = data_sorted[(data_sorted['Date'] == current_date) & (
#                     data_sorted['Company'] == stock)]['Open'].values[0]
#                 cash += sell_price * info['shares']
#                 print(
#                     f"Sold {stock} for ${sell_price:.2f}/share, total ${sell_price * info['shares']:.2f}. New cash balance: ${cash:.2f}")
#             del portfolio[current_date]  # Remove sold stocks from portfolio

#         # Check for 30 Day Sell Signals and queue them for next trading day
#         for index, row in today_data.iterrows():
#             if row['30 Day Sell Signal'] == 1:
#                 next_sell_day = unique_dates[min(
#                     today_index + 1, len(unique_dates)-1)]
#                 if next_sell_day not in sell_signals:
#                     sell_signals[next_sell_day] = []
#                 sell_signals[next_sell_day].append(row['Company'])

#         # Buy condition: Check for buy signals and place orders for the next day
#         for index, row in today_data.iterrows():
#             if row['30 Day Buy Signal'] == 1 and today_index + 14 < len(unique_dates):
#                 next_day = unique_dates[today_index + 14]
#                 amount_to_spend = cash * 0.5
#                 next_day_data = data_sorted[(data_sorted['Date'] == next_day) & (
#                     data_sorted['Company'] == row['Company'])]
#                 if not next_day_data.empty:
#                     next_day_open_price = next_day_data['Open'].values[0]
#                     shares_to_buy = amount_to_spend / next_day_open_price
#                     cash -= amount_to_spend
#                     # Determine sell date, 30 days later
#                     sell_date = unique_dates[min(
#                         today_index + 22, len(unique_dates)-1)]

#                     # Track buy in portfolio for future selling
#                     if sell_date not in portfolio:
#                         portfolio[sell_date] = {}
#                     portfolio[sell_date][row['Company']] = {
#                         'shares': shares_to_buy, 'buy_price': next_day_open_price}

#                     print(
#                         f"Buy Order for {row['Company']} will be placed on {next_day.strftime('%Y-%m-%d')} at ${next_day_open_price:.2f} for {shares_to_buy:.2f} shares. Amount Spent: ${amount_to_spend:.2f}. Will sell on: {sell_date.strftime('%Y-%m-%d')}")
#                     print(f"Will sell on: {sell_date.strftime('%Y-%m-%d')}")
#                 else:
#                     print(
#                         f"No trading data available for {row['Company']} on {next_day.strftime('%Y-%m-%d')}")

#     # Update portfolio value for current day
#         current_portfolio_value = cash  # Include current cash in portfolio value
#         for _, stocks in portfolio.items():
#             for stock, info in stocks.items():
#                 stock_data_today = data_sorted[(data_sorted['Date'] == current_date) & (
#                     data_sorted['Company'] == stock)]
#                 if not stock_data_today.empty:
#                     # If the stock was bought, use its last known price for value calculation
#                     last_price = stock_data_today['Open'].iloc[0]
#                     current_portfolio_value += last_price * info['shares']
#         portfolio_value_records.append(
#             {'Date': current_date, 'Value': current_portfolio_value})

#     # Convert the records list to a DataFrame for plotting
#     portfolio_values_df = pd.DataFrame(portfolio_value_records)

#     # Plot the portfolio value over time
#     st.line_chart(portfolio_values_df.set_index('Date'))

#     # Final calculations and displaying results...
#     final_portfolio_value = portfolio_values_df['Value'].iloc[-1] if not portfolio_values_df.empty else initial_cash
#     roi = ((final_portfolio_value - initial_cash) / initial_cash) * 100

#     st.write("Simulation has ended.")
#     st.write(f"Initial Investment: ${initial_cash:.2f}")
#     st.write(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
#     st.write(f"Current Cash: ${cash:.2f}")
#     st.write(f"ROI: {roi:.2f}%")

