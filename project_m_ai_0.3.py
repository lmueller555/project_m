import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Load the dataset
data = pd.read_csv(
    'C:\\Users\\lmueller\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv')

# Ensure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date', inplace=True)

# Remove numbers, periods, and trailing spaces from the 'Company' column
data['Company'] = data['Company'].str.replace(
    r'\d+\.', '', regex=True).str.strip()


def identify_pairs(correlation_matrix, threshold=0.95):
    pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > threshold:
                pairs.append(
                    (correlation_matrix.columns[i], correlation_matrix.columns[j]))
    return pairs


def calculate_normal_ratios(pivoted_data, pair, historical_window=60):
    daily_ratios = pivoted_data[pair[0]] / pivoted_data[pair[1]]
    normal_ratio = daily_ratios[-historical_window:].mean()
    return normal_ratio


# Pivot the data to get closing prices of each stock in columns
pivoted_data = data.pivot(index='Date', columns='Company', values='Close/Last')

# Calculating the correlation matrix for the closing prices of stocks
correlation_matrix = pivoted_data.corr()

pairs = identify_pairs(correlation_matrix, threshold=0.95)
normal_ratios = {pair: calculate_normal_ratios(
    pivoted_data, pair) for pair in pairs}

for pair in pairs:
    print(pair)
    # print(f"Pair: {pair}, Normal Ratio: {normal_ratios[pair]}")
    if pair[0] not in data['Company'].unique() or pair[1] not in data['Company'].unique():
        print(f"Data for pair {pair} is not available in the dataset.")


def reformat_pair_names(pair):
    stock1, stock2 = pair
    # Remove numbers, periods, and trailing spaces (if necessary)
    stock1 = re.sub(r'\d+\.', '', stock1).strip()
    stock2 = re.sub(r'\d+\.', '', stock2).strip()
    return (stock1, stock2)


# Now apply this function to each pair in the pairs list
formatted_pairs = [reformat_pair_names(pair) for pair in pairs]


def initialize_portfolio(initial_capital):
    return {'cash': initial_capital, 'stocks': {}}


def buy_stock(portfolio, stock, price):
    quantity = min(MAX_TRADE_QUANTITY, portfolio['cash'] // price)
    if quantity > 0 and stock not in portfolio['stocks']:
        portfolio['stocks'][stock] = {'quantity': 0, 'average_buying_price': 0}

    if quantity > 0:
        total_cost = price * quantity
        total_quantity = portfolio['stocks'][stock]['quantity'] + quantity
        average_price = ((portfolio['stocks'][stock]['average_buying_price'] *
                          portfolio['stocks'][stock]['quantity']) + total_cost) / total_quantity

        # Update portfolio
        portfolio['cash'] -= total_cost
        portfolio['stocks'][stock]['quantity'] = total_quantity
        portfolio['stocks'][stock]['average_buying_price'] = average_price


def sell_stock(portfolio, stock, price):
    if stock in portfolio['stocks']:
        quantity = min(MAX_TRADE_QUANTITY,
                       portfolio['stocks'][stock]['quantity'])
        if quantity > 0:
            # Update portfolio
            portfolio['cash'] += price * quantity
            portfolio['stocks'][stock]['quantity'] -= quantity

            if portfolio['stocks'][stock]['quantity'] == 0:
                del portfolio['stocks'][stock]


def momentum_trading(row, portfolio):
    momentum_threshold = 0.025  # 5%
    stock = row['Company'].strip()
    price = row['Close/Last']
    prediction = row['Prediction']

    if prediction > price * (1 + momentum_threshold):
        buy_stock(portfolio, stock, price)
    elif prediction < price * (1 - momentum_threshold):
        sell_stock(portfolio, stock, price)


def mean_reversion(row, portfolio):
    mean_rev_threshold = 0.025  # 5%
    stock = row['Company'].strip()
    price = row['Close/Last']
    if row['Close/Last'] < row['30d MA'] * (1 - mean_rev_threshold):
        buy_stock(portfolio, stock, price)
    elif row['Close/Last'] > row['30d MA'] * (1 + mean_rev_threshold):
        sell_stock(portfolio, stock, price)


def swing_trading(row, portfolio):
    swing_threshold = 0.05  # Example threshold
    stock = row['Company'].strip()
    price = row['Close/Last']
    if row['Prediction'] > price * (1 + swing_threshold):
        buy_stock(portfolio, stock, price)
    elif row['Prediction'] < price * (1 - swing_threshold):
        sell_stock(portfolio, stock, price)


def pair_trading(aggregated_day_data, portfolio, pairs, normal_ratios, entry_threshold=0.05, exit_threshold=0.03):
    for pair in pairs:
        stock1, stock2 = pair

        if stock1 in aggregated_day_data and stock2 in aggregated_day_data:
            stock1_data = aggregated_day_data[stock1]
            stock2_data = aggregated_day_data[stock2]
            current_ratio = stock1_data['Close/Last'] / \
                stock2_data['Close/Last']
            normal_ratio = normal_ratios[pair]

            if current_ratio < normal_ratio * (1 - entry_threshold):
                # Long on stock1, short on stock2
                if portfolio['cash'] >= stock1_data['Close/Last']:
                    quantity = 1  # Assuming buying one unit of stock
                    portfolio['cash'] -= stock1_data['Close/Last']
                    if stock1 not in portfolio['stocks']:
                        portfolio['stocks'][stock1] = {
                            'quantity': 0, 'average_buying_price': stock1_data['Close/Last']}
                    portfolio['stocks'][stock1]['quantity'] += quantity

            elif current_ratio > normal_ratio * (1 + entry_threshold):
                # Short on stock1, long on stock2
                if portfolio['cash'] >= stock2_data['Close/Last']:
                    quantity = 1  # Assuming buying one unit of stock
                    portfolio['cash'] -= stock2_data['Close/Last']
                    if stock2 not in portfolio['stocks']:
                        portfolio['stocks'][stock2] = {
                            'quantity': 0, 'average_buying_price': stock2_data['Close/Last']}
                    portfolio['stocks'][stock2]['quantity'] += quantity

            if normal_ratio * (1 - exit_threshold) < current_ratio < normal_ratio * (1 + exit_threshold):
                # Implement logic to close the trade here
                # This includes selling long positions and covering short positions
                pass
        else:
            missing_stocks = [s for s in (
                stock1, stock2) if s not in aggregated_day_data]
            print(
                f"Data for stocks {missing_stocks} in pair {pair} not available on {aggregated_day_data.name}")


# Define thresholds
entry_threshold = 0.05  # 5% deviation from normal ratio for entering trades
exit_threshold = 0.03   # 3% deviation from normal ratio for exiting trades


def calculate_portfolio_value(portfolio, current_prices):
    total_stock_value = 0
    for stock, info in portfolio['stocks'].items():
        if stock in current_prices:
            total_stock_value += current_prices[stock] * info['quantity']
        else:
            print(
                f"Debug: {stock} not found in current_prices. Available keys: {list(current_prices.keys())}")
    return portfolio['cash'] + total_stock_value


def calculate_roi(initial_capital, current_value):
    return (current_value - initial_capital) / initial_capital


def calculate_maximum_drawdown(portfolio_values):
    peak = max(portfolio_values)
    trough = min(portfolio_values)
    return (peak - trough) / peak


def adjust_parameters_based_on_roi(roi, last_roi, increment_factor=0.01):
    global MAX_TRADE_QUANTITY
    # Adjust MAX_TRADE_QUANTITY
    if roi > last_roi:  # If ROI has increased
        MAX_TRADE_QUANTITY += int(MAX_TRADE_QUANTITY * increment_factor)
    elif roi < last_roi:  # If ROI has decreased
        MAX_TRADE_QUANTITY = max(
            1, int(MAX_TRADE_QUANTITY * (1 - increment_factor)))

    # Add similar logic for adjusting strategy thresholds


# Initialize variables
MAX_TRADE_QUANTITY = 10
last_roi = 0
initial_capital = 100000
portfolio = initialize_portfolio(initial_capital)
daily_portfolio_values_last_gen = []
portfolio_values = []
evaluation_interval = 30  # Days

# Extract unique trading days and sort them
unique_trading_days = pd.to_datetime(data['Date'].unique())
unique_trading_days = np.sort(unique_trading_days)

# Number of generations for the simulation
num_generations = 1
generation_rois = []

for generation in range(num_generations):
    print(f"\nStarting Generation {generation + 1}")
    portfolio = initialize_portfolio(initial_capital)
    portfolio_values = []
    last_roi = 0
    current_prices = {}

    for trading_day in unique_trading_days:
        trading_day = pd.Timestamp(trading_day)
        print(f"Processing trading day: {trading_day.strftime('%Y-%m-%d')}")

        # Filter data up to the current trading day for correlation calculation
        historical_data = data[data['Date'] <= trading_day]
        pivoted_historical_data = historical_data.pivot(
            index='Date', columns='Company', values='Close/Last')

        # Recalculate correlation matrix
        current_correlation_matrix = pivoted_historical_data.corr()

        # Identify pairs and calculate normal ratios using the current correlation matrix
        current_pairs = identify_pairs(
            current_correlation_matrix, threshold=0.95)
        formatted_current_pairs = [
            reformat_pair_names(pair) for pair in current_pairs]
        current_normal_ratios = {pair: calculate_normal_ratios(
            pivoted_historical_data, pair) for pair in formatted_current_pairs}

        daily_data = data[data['Date'] == trading_day]
        if daily_data.empty:
            print(
                f"No data for trading day: {trading_day.strftime('%Y-%m-%d')}")
            continue

        # Aggregate data for the current trading day
        aggregated_day_data = {row['Company'].strip(
        ): row for _, row in daily_data.iterrows()}

        for _, row in daily_data.iterrows():
            stock_name = row['Company'].strip()
            current_prices[stock_name] = row['Close/Last']

            # Apply trading strategies
            momentum_trading(row, portfolio)
            mean_reversion(row, portfolio)
            swing_trading(row, portfolio)

        # Apply pair trading using the updated pairs and ratios
        pair_trading(aggregated_day_data, portfolio, formatted_current_pairs,
                     current_normal_ratios, entry_threshold, exit_threshold)

        # Calculate current portfolio value
        current_value = calculate_portfolio_value(portfolio, current_prices)
        portfolio_values.append(current_value)

        if len(portfolio_values) % evaluation_interval == 0:
            current_roi = calculate_roi(initial_capital, current_value)
            max_drawdown = calculate_maximum_drawdown(
                portfolio_values[-evaluation_interval:])
            print(
                f"Day {trading_day.strftime('%Y-%m-%d')}: ROI: {current_roi:.2%}, Max Drawdown: {max_drawdown:.2%}")
            adjust_parameters_based_on_roi(current_roi, last_roi)
            last_roi = current_roi

    final_roi = calculate_roi(initial_capital, portfolio_values[-1])
    generation_rois.append(final_roi)

    if generation == num_generations - 1:  # If it's the last generation
        daily_portfolio_values_last_gen = portfolio_values.copy()

    total_portfolio_value = portfolio_values[-1]
    total_profit = total_portfolio_value - initial_capital
    print(f"End of Generation {generation + 1}: Final ROI: {final_roi:.2%}")
    print(f"Final Portfolio for Generation {generation + 1}: {portfolio}")
    print(
        f"Total Portfolio Value at End of Generation {generation + 1}: ${total_portfolio_value:,.2f}")
    print(
        f"Total Profit at End of Generation {generation + 1}: ${total_profit:,.2f}")


# After the loop over generations
final_portfolio = portfolio

# Extracting stock names and their quantities from the final portfolio
stock_names = list(final_portfolio['stocks'].keys())
quantities = [final_portfolio['stocks'][stock]['quantity']
              for stock in stock_names]

# Plotting daily portfolio values for the last generation
plt.figure(figsize=(14, 7))
plt.plot(daily_portfolio_values_last_gen)
plt.title('Daily Portfolio Value')
plt.xlabel('Day')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()

# Creating the bar graph
plt.figure(figsize=(14, 7))
plt.bar(stock_names, quantities, color='blue')
plt.xlabel('Stocks')
plt.ylabel('Quantity')
plt.title('Portfolio Composition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

