import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tqdm.keras import TqdmCallback
import random
import matplotlib.pyplot as plt

# File path for the Excel file
file_path = 'C:\\Users\\lmueller\\Desktop\\Project M\\Project M Data Processed.xlsx'

# Reading and preprocessing data
dataset = pd.read_excel(file_path)
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.set_index('Date', inplace=True)

# Using 'Close/Last' for prediction and scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset['Close/Last'].values.reshape(-1, 1))

# Creating a dataset suitable for time series prediction


def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


# Splitting the dataset into training and testing sets
time_step = 100
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model with a progress bar
epochs = 5
batch_size = 64
model.fit(X, y, epochs=epochs, batch_size=batch_size,
          verbose=0, callbacks=[TqdmCallback(verbose=1)])

# Define a function for the trading simulation with LSTM model


def trading_simulation(dataset, buy_threshold, sell_threshold, take_profit_threshold, stop_loss_threshold, deviation_threshold, model, scaler):
    cash = 100000  # Initial cash
    shares_held = 0  # Initial shares
    buy_price = 0
    investment_values = []  # Track investment value over time
    trade_count = 0  # Count the number of trades

    for index in range(len(dataset) - time_step):
        current_price = dataset.iloc[index + time_step]['Close/Last']
        previous_data = scaled_data[index:index + time_step]
        predicted_price = scaler.inverse_transform(
            model.predict(previous_data.reshape(1, time_step, 1)))[0][0]

        ma_200d = dataset.iloc[index + time_step]['200d MA']
        deviation = (current_price - ma_200d) / ma_200d

        # Check for take-profit or stop-loss conditions
        if shares_held > 0:
            if current_price >= buy_price * take_profit_threshold or current_price <= buy_price * stop_loss_threshold:
                cash += shares_held * current_price
                shares_held = 0
                trade_count += 1

        # Buy or sell based on the deviation and predicted price
        if deviation < -deviation_threshold and predicted_price > current_price * buy_threshold and cash >= current_price:
            shares_to_buy = cash // current_price
            cash -= shares_to_buy * current_price
            shares_held += shares_to_buy
            buy_price = current_price  # Update the buy_price
            trade_count += 1
        elif deviation > deviation_threshold and predicted_price < current_price * sell_threshold and shares_held > 0:
            cash += shares_held * current_price
            shares_held = 0
            trade_count += 1

        investment_values.append(cash + shares_held * current_price)

    return investment_values, cash + shares_held * dataset.iloc[-1]['Close/Last'] - 100000, trade_count


# Running multiple generations with take-profit and stop-loss
generations = 1000
best_deviation_threshold = 0.09  # Initial deviation threshold
best_profit = -float('inf')
best_investment_values = []
best_trade_count = 0
best_parameters = {}

for _ in range(generations):
    # Randomly adjust thresholds and parameters
    buy_threshold = random.uniform(0.95, 1.05)
    sell_threshold = random.uniform(0.95, 1.05)
    take_profit_threshold = random.uniform(0.95, 1.05)
    stop_loss_threshold = random.uniform(0.85, 0.95)
    deviation_threshold = best_deviation_threshold * random.uniform(0.95, 1.05)

    # Run the simulation
    investment_values, profit, trade_count = trading_simulation(
        dataset, buy_threshold, sell_threshold, take_profit_threshold, stop_loss_threshold, deviation_threshold, model, scaler)

    # Update best performing generation
    if profit > best_profit:
        best_profit = profit
        best_parameters = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'take_profit_threshold': take_profit_threshold,
            'stop_loss_threshold': stop_loss_threshold,
            'deviation_threshold': deviation_threshold
        }
        best_investment_values = investment_values
        best_trade_count = trade_count

# Output the results of the best generation
print(
    f"Best Generation - Profit: {best_profit}, Trades: {best_trade_count}, Parameters: {best_parameters}")

# Plotting the investment value over time for the best generation
plt.figure(figsize=(12, 6))
plt.plot(best_investment_values, label='Investment Value')
plt.title('Best Generation AI Investment Value Over Time')
plt.xlabel('Days')
plt.ylabel('Investment Value in $')
plt.legend()
plt.grid(True)
plt.show()


# C:\\Users\\lmueller\\Desktop\\Project M\\Project M Data Processed.xlsx
