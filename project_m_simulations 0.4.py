import random
import pandas as pd
import matplotlib.pyplot as plt

class TradingAI:
    def __init__(self, initial_capital, dataset):
        self.initial_capital = initial_capital
        self.dataset = dataset
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.shares_held = {company: 0 for company in self.dataset['Company'].unique()}
        self.total_value = self.initial_capital
        self.trade_count = 0
        print("AI reset: Starting new simulation.")

    def evaluate_strategy(self, day):
        self.total_value = self.cash + sum(self.shares_held[company] * self.dataset.loc[day, 'Close/Last'] for company in self.shares_held)
        print(f"Day {day}: Total Portfolio Value: {self.total_value}, Cash: {self.cash}")
        return self.total_value

    def trade(self, day, parameters):
        for company in self.dataset['Company'].unique():
            row = self.dataset.loc[day]
            if row['Company'] != company:
                continue

            current_price = row['Close/Last']
            ma_30d = row['30d MA']
            ma_200d = row['200d MA']

            # Check for crossover and deviation conditions
            crossover = ma_30d > ma_200d
            deviation = (current_price - ma_200d) / ma_200d

            # Decision logic
            if deviation < parameters['lower_deviation'] and crossover and self.cash >= current_price:
                shares_to_buy = self.cash // current_price
                self.cash -= shares_to_buy * current_price
                self.shares_held[company] += shares_to_buy
                self.trade_count += 1
                print(f"Day {day}: Bought {shares_to_buy} shares of {company} at {current_price}. Cash left: {self.cash}, Shares Held: {self.shares_held[company]}")
            elif deviation > parameters['upper_deviation'] and not crossover and self.shares_held[company] > 0:
                self.cash += self.shares_held[company] * current_price
                self.shares_held[company] = 0
                self.trade_count += 1
                print(f"Day {day}: Sold all shares of {company} at {current_price}. Cash: {self.cash}, Shares Held: {self.shares_held[company]}")

    def run_simulation(self, parameters):
        for day in range(len(self.dataset)):
            self.trade(day, parameters)
            if day % 30 == 0:  # Evaluate every 30 days
                self.evaluate_strategy(day)

        return self.evaluate_strategy(len(self.dataset) - 1) - self.initial_capital

# Load dataset
dataset = pd.read_csv('C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project M Cleaned Data.csv')

# Initialize the AI
ai = TradingAI(initial_capital=100000, dataset=dataset)

# Run multiple generations with adjusted strategy
generations = 100
best_parameters = {'lower_deviation': -0.05, 'upper_deviation': 0.05}
best_profit = -float('inf')

for i in range(generations):
    # Randomly adjust parameters
    parameters = {
        'lower_deviation': random.uniform(-0.1, 0),
        'upper_deviation': random.uniform(0, 0.1)
    }

    # Run the simulation
    ai.reset()
    profit = ai.run_simulation(parameters)

    # Update best performing generation
    if profit > best_profit:
        best_profit = profit
        best_parameters = parameters

    # Print information about each generation
    print(f"Generation {i + 1}: Profit: {profit}, Parameters: {parameters}")

# Output the results of the best generation
print(f"Best Generation - Profit: {best_profit}, Parameters: {best_parameters}")

# file_path = 'C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project M Data Processed.xlsx'