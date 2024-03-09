import streamlit as st
import pandas as pd

# Streamlit app title
st.title('Project M Portfolio Simulation')

# Load the dataset
file_path = 'https://raw.githubusercontent.com/lmueller555/project_m/main/Updated_Dataset_with_Signals_Ranked.csv'
df = pd.read_csv(file_path)

# Convert 'Date' to datetime and sort the dataframe
df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# Define the minimum and maximum dates available in the dataset for date input limits
min_date = df_sorted['Date'].min()
max_date = df_sorted['Date'].max()

# User inputs for the simulation
start_date = st.date_input('Start date', value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input('End date', value=max_date, min_value=min_date, max_value=max_date)
initial_investment = st.number_input('Initial Investment Amount', min_value=0, value=50000, step=1000)
monthly_contribution = st.number_input('Monthly Contribution Amount', min_value=0, value=3000, step=100)

# Convert the start_date and end_date to pd.Timestamp to match dtype for comparison
start_date_pd = pd.Timestamp(start_date)
end_date_pd = pd.Timestamp(end_date)

# Filter the dataframe based on the selected date range
df_filtered = df_sorted[(df_sorted['Date'] >= start_date_pd) & (df_sorted['Date'] <= end_date_pd)]
date_index = df_filtered['Date'].unique()

# Initialize variables for the backtest
cash_available = initial_investment
portfolio = []  # To store active positions
trades = []  # To log the outcome of each trade
portfolio_values = []  # To store the total portfolio value for plotting
contribution_counter = 0  # Counter to track trading days for contributions
line_chart_placeholder = st.empty()
bar_chart_placeholder = st.empty()

# Function to calculate portfolio value
def calculate_portfolio_value(portfolio, current_date):
    value = 0
    for position in portfolio:
        current_price_data = df_filtered[(df_filtered['Company'] == position['company']) & (df_filtered['Date'] == current_date)]
        if not current_price_data.empty:
            current_price = current_price_data.iloc[0]['Close/Last']
            value += position['shares_bought'] * current_price
    return value

# Backtesting logic
if st.button('Run Simulation'):
    total_days = len(date_index)
    progress_bar = st.progress(0)
    status_text = st.empty()  # Placeholder for dynamic text
    
    for i, current_date in enumerate(date_index):
        # Update the progress bar and status text
        progress = (i + 1) / total_days
        progress_bar.progress(progress)
        status_text.text(f"Simulating trading day {i + 1} of {total_days} ({progress * 100:.2f}%)")
        
        contribution_counter += 1
        if contribution_counter == 22:
            cash_available += monthly_contribution
            contribution_counter = 0

        daily_data = df_filtered[df_filtered['Date'] == current_date]
        for _, row in daily_data.iterrows():
            if row['30 Day Buy Signal'] == 1 and cash_available > 0:
                if i + 1 < len(date_index):
                    next_day_data = df_filtered[(df_filtered['Company'] == row['Company']) & (df_filtered['Date'] == date_index[i + 1])]
                    if not next_day_data.empty:
                        next_open_price = next_day_data.iloc[0]['Open']
                        max_shares_to_buy = (cash_available * 0.5) / next_open_price
                        if max_shares_to_buy >= 1:
                            invest_amount = max_shares_to_buy * next_open_price
                            cash_available -= invest_amount
                            shares_bought = invest_amount / next_open_price
                            # Update portfolio dictionary
                            if row['Company'] in portfolio:
                                portfolio[row['Company']] += shares_bought
                            else:
                                portfolio[row['Company']] = shares_bought
                            sell_date_index = i + 32 if i + 32 < len(date_index) else -1
                            sell_date = date_index[sell_date_index]
                            # Append to trades for later removal logic
                            trades.append({'company': row['Company'], 'sell_date': sell_date, 'shares_sold': shares_bought})

        # Remove sold positions from the portfolio
        for trade in trades:
            if current_date == trade['sell_date']:
                portfolio[trade['company']] -= trade['shares_sold']
                if portfolio[trade['company']] <= 0:
                    del portfolio[trade['company']]

        # Update line chart with total portfolio value
        portfolio_value = sum([calculate_portfolio_value({company: shares}, current_date) for company, shares in portfolio.items()])
        total_value = cash_available + portfolio_value
        portfolio_values.append(total_value)
        line_chart_data = pd.DataFrame({
            'Date': date_index[:i+1],
            'Total Portfolio Value': portfolio_values
        }).set_index('Date')
        line_chart_placeholder.line_chart(line_chart_data)

        # Update bar chart with currently held stocks
        bar_chart_data = pd.DataFrame({
            'Company': list(portfolio.keys()),
            'Shares': list(portfolio.values())
        })
        bar_chart_placeholder.bar_chart(bar_chart_data.set_index('Company'))

    progress_bar.empty()

    # Adjustments for total contributions made
    total_contributions = initial_investment + (monthly_contribution * (len(date_index) // 22))
    final_portfolio_value = cash_available + calculate_portfolio_value(portfolio, date_index[-1])
    roi = ((final_portfolio_value - total_contributions) / total_contributions) * 100
    win_rate = (sum(trades) / len(trades)) * 100 if trades else 0

    # Displaying results
    st.write(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    st.write(f"Total Contributions: ${total_contributions:,.2f}")
    st.write(f"ROI: {roi:.2f}%")
    st.write(f"Trading Win Rate: {win_rate:.2f}%")
