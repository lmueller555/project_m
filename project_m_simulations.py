import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt

# File path for the Excel file
file_path = 'C:\\Users\\lmueller\\Desktop\\Project M\\Project M Data Processed.xlsx'

# Reading the data from the Excel file
dataset = pd.read_excel(file_path)
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.set_index('Date', inplace=True)

# Generate features for the predictive model
dataset['Close/Last Shifted'] = dataset['Close/Last'].shift(1)
dataset.dropna(inplace=True)  # Drop NaN values created by the shift operation

# Define a function for the trading simulation with take-profit, stop-loss, and trade count


def trading_simulation(dataset, buy_threshold, sell_threshold, take_profit_threshold, stop_loss_threshold, model):
    cash = 100000  # Initial cash
    shares_held = 0  # Initial shares
    buy_price = 0  # Price at which shares were bought
    investment_values = []  # Track investment value over time
    trade_count = 0  # Count the number of trades

    for index, row in dataset.iterrows():
        current_price = row['Close/Last']
        predicted_next_close = model.predict([[row['Close/Last Shifted']]])[0]

        # Check for take-profit or stop-loss conditions
        if shares_held > 0:
            if current_price >= buy_price * take_profit_threshold or current_price <= buy_price * stop_loss_threshold:
                cash += shares_held * current_price
                shares_held = 0
                trade_count += 1

        # Buy or sell based on the predicted price
        if predicted_next_close > current_price * buy_threshold and cash >= current_price:
            shares_to_buy = cash // current_price
            cash -= shares_to_buy * current_price
            shares_held += shares_to_buy
            buy_price = current_price
            trade_count += 1
        elif predicted_next_close < current_price * sell_threshold and shares_held > 0:
            cash += shares_held * current_price
            shares_held = 0
            trade_count += 1

        investment_values.append(cash + shares_held * current_price)

    return investment_values, cash + shares_held * dataset.iloc[-1]['Close/Last'] - 100000, trade_count


# Create a simple linear regression model to predict next day's close price
model = LinearRegression()

# Running multiple generations with take-profit and stop-loss
generations = 1000
best_buy_threshold = 1.01
best_sell_threshold = 0.99
best_take_profit = 1.05
best_stop_loss = 0.85
best_profit = -float('inf')
best_investment_values = []
best_trade_count = 0
best_model = None

for _ in range(generations):
    # Randomly adjust thresholds
    buy_threshold = best_buy_threshold * random.uniform(0.95, 1.05)
    sell_threshold = best_sell_threshold * random.uniform(0.95, 1.05)
    take_profit_threshold = best_take_profit * random.uniform(0.95, 1.05)
    stop_loss_threshold = best_stop_loss * random.uniform(0.85, 0.95)

    # Train the predictive model
    X_train = dataset[['Close/Last Shifted']]
    y_train = dataset['Close/Last']
    model.fit(X_train, y_train)

    # Run the simulation
    investment_values, profit, trade_count = trading_simulation(
        dataset, buy_threshold, sell_threshold, take_profit_threshold, stop_loss_threshold, model)

    # Update best performing generation
    if profit > best_profit:
        best_profit = profit
        best_buy_threshold = buy_threshold
        best_sell_threshold = sell_threshold
        best_take_profit = take_profit_threshold
        best_stop_loss = stop_loss_threshold
        best_investment_values = investment_values
        best_trade_count = trade_count
        best_model = model

# Output the results of the best generation
print(f"Best Generation - Profit: {best_profit}, Trades: {best_trade_count}, Buy Threshold: {best_buy_threshold}, Sell Threshold: {best_sell_threshold}, Take Profit: {best_take_profit}, Stop Loss: {best_stop_loss}")

# Plotting the investment value over time for the best generation
plt.figure(figsize=(12, 6))
plt.plot(best_investment_values, label='Investment Value')
plt.title('Best Generation AI Investment Value Over Time')
plt.xlabel('Days')
plt.ylabel('Investment Value in $')
plt.legend()
plt.grid(True)
plt.show()
