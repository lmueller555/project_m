import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
# file_path = r'C:\Users\lmueller\Desktop\Project M\Updated_Dataset_with_Indicators_Test2.csv'
file_path = r'C:\Users\lmueller\Desktop\Project M\Updated_Dataset_with_Signals_Ranked.csv'
df = pd.read_csv(file_path)

# Convert 'Date' to datetime and sort the dataframe
df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# Create a date index
date_index = df_sorted['Date'].unique()

# Initialize variables for the backtest
initial_investment = 50000
cash_available = initial_investment
portfolio = []  # To store active positions
trades = []  # To log the outcome of each trade
portfolio_values = []  # To store the total portfolio value for plotting

# Function to calculate portfolio value


def calculate_portfolio_value(portfolio, current_date):
    value = 0
    for position in portfolio:
        current_price_data = df_sorted[(df_sorted['Company'] == position['company']) & (
            df_sorted['Date'] == current_date)]
        if not current_price_data.empty:
            current_price = current_price_data.iloc[0]['Close/Last']
            value += position['shares_bought'] * current_price
    return value


# Backtesting logic
for i, current_date in enumerate(date_index):
    daily_data = df_sorted[df_sorted['Date'] == current_date]
    for _, row in daily_data.iterrows():
        if row['30 Day Buy Signal'] == 1 and cash_available > 0:
            if i + 1 < len(date_index):
                next_day_data = df_sorted[(df_sorted['Company'] == row['Company']) & (
                    df_sorted['Date'] == date_index[i + 1])]
                if not next_day_data.empty:
                    next_open_price = next_day_data.iloc[0]['Open']
                    max_shares_to_buy = (
                        cash_available * 0.5) / next_open_price
                    if max_shares_to_buy >= 1:
                        invest_amount = max_shares_to_buy * next_open_price
                        cash_available -= invest_amount
                        shares_bought = invest_amount / next_open_price
                        sell_date_index = i + 22 if i + \
                            22 < len(date_index) else -1
                        sell_date = date_index[sell_date_index]
                        portfolio.append({
                            'company': row['Company'],
                            'sell_date': sell_date,
                            'shares_bought': shares_bought,
                            'buy_price': next_open_price
                        })
    for position in portfolio[:]:
        if current_date == position['sell_date']:
            sell_day_data = df_sorted[(df_sorted['Company'] == position['company']) & (
                df_sorted['Date'] == position['sell_date'])]
            if not sell_day_data.empty:
                sell_price = sell_day_data.iloc[0]['Close/Last']
                profit = (
                    sell_price - position['buy_price']) * position['shares_bought']
                cash_available += position['shares_bought'] * sell_price
                portfolio.remove(position)
                trades.append(profit > 0)
    portfolio_value = calculate_portfolio_value(portfolio, current_date)
    total_value = cash_available + portfolio_value
    portfolio_values.append(total_value)
    print(f"Date: {current_date.strftime('%Y-%m-%d')}, Portfolio Value: ${total_value:,.2f}, Available Cash: ${cash_available:,.2f}")

# Final calculations
final_portfolio_value = cash_available + \
    calculate_portfolio_value(portfolio, date_index[-1])
roi = ((final_portfolio_value - initial_investment) / initial_investment) * 100
win_rate = (sum(trades) / len(trades)) * 100 if trades else 0
print(f"\nFinal Portfolio Value: ${final_portfolio_value:,.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Trading Win Rate: {win_rate:.2f}%")

# Plotting the total portfolio value over time
plt.figure(figsize=(10, 6))
plt.plot(date_index, portfolio_values,
         label='Total Portfolio Value', color='blue')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Total Portfolio Value ($)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
