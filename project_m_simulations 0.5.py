import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class TradingAI:
    def __init__(self, initial_capital, dataset):
        self.initial_capital = initial_capital
        self.dataset = dataset
        self.total_days = len(dataset['Date'].unique())
        self.correlations = self.calculate_rolling_correlations()

    def reset(self, generation_number=None):
        self.cash = self.initial_capital
        self.shares_held = {
            company: 0 for company in self.dataset['Company'].unique()}
        self.purchase_prices = {
            company: 0 for company in self.dataset['Company'].unique()}
        self.total_value = self.initial_capital
        self.trade_count = 0
        if generation_number is not None:
            print(f"AI reset for Generation {generation_number}.")
        else:
            print("AI initialized with starting capital.")

    def calculate_rolling_correlations(self, window_size=30):
        correlations = {}
        # Ensure data is sorted by date
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        sorted_df = self.dataset.sort_values('Date')
        unique_dates = sorted_df['Date'].unique()

        for start_idx in range(0, len(unique_dates), window_size):
            end_idx = min(start_idx + window_size, len(unique_dates))
            date_range = unique_dates[start_idx:end_idx]
            window_df = sorted_df[sorted_df['Date'].isin(date_range)]
            pivot_df = window_df.pivot(
                index='Date', columns='Company', values='Close/Last')
            corr_matrix = pivot_df.corr()
            correlations[date_range[-1]] = corr_matrix
        return correlations

    def trade_using_correlations(self, current_date, shares_held):
        # This is a simplified example of how you might use correlations
        correlated_stocks = self.correlations[current_date].idxmax()
        # Sell shares of stocks that are highly correlated with stocks that are dropping
        for company in shares_held:
            if shares_held[company] > 0:
                correlated_company = correlated_stocks[company]
                if correlated_company in shares_held and shares_held[correlated_company] > 0:
                    self.sell_stock(
                        correlated_company, self.dataset[self.dataset['Company'] == correlated_company]['Close/Last'].iloc[-1], "Correlated Drop")

    def evaluate_strategy(self, day):
        day_data = self.dataset[self.dataset['Date'] == day]
        self.total_value = self.cash + sum(
            self.shares_held[company] * day_data[day_data['Company']
                                                 == company]['Close/Last'].iloc[0]
            for company in self.shares_held)
        return self.total_value

    def trade(self, day, parameters):
        day_data = self.dataset[self.dataset['Date'] == day]
        for index, row in day_data.iterrows():
            company = row['Company']
            current_price = row['Close/Last']
            ma_30d = row['30d MA']
            ma_200d = row['200d MA']

            crossover = ma_30d > ma_200d
            deviation = (current_price - ma_200d) / ma_200d

            max_investment_per_stock = self.cash * \
                parameters['max_allocation_per_stock']

            if self.shares_held[company] > 0:
                purchase_price = self.purchase_prices[company]
                if current_price <= purchase_price * (1 - parameters['stop_loss']):
                    self.sell_stock(company, current_price, "Stop Loss")
                elif current_price >= purchase_price * (1 + parameters['take_gain']):
                    self.sell_stock(company, current_price, "Take Gain")

            if deviation < parameters['lower_deviation'] and crossover and self.cash >= current_price:
                investment_amount = min(self.cash, max_investment_per_stock)
                shares_to_buy = investment_amount // current_price
                self.cash -= shares_to_buy * current_price
                self.shares_held[company] += shares_to_buy
                self.purchase_prices[company] = current_price
                self.trade_count += 1
                print(
                    f"Day {day}: Bought {shares_to_buy} shares of {company} at {current_price}. Cash left: {self.cash}, Shares Held: {self.shares_held[company]}")

    def sell_stock(self, company, current_price, reason):
        self.cash += self.shares_held[company] * current_price
        self.shares_held[company] = 0
        print(
            f"Sold all shares of {company} at {current_price} due to {reason}. Cash: {self.cash}, Shares Held: 0")
        self.trade_count += 1

    def run_simulation(self, parameters, generation_number):
        evaluation_interval = 30  # Days between each evaluation
        last_evaluation_value = self.initial_capital
        adjustment_factor = 0.05  # Factor to adjust the parameters

        # Iterate over the unique dates in the dataset
        for day_index, day in enumerate(self.dataset['Date'].unique()):
            self.trade(day, parameters)

            # Perform evaluation every 'evaluation_interval' days
            if day_index % evaluation_interval == 0 or day_index == self.total_days - 1:
                current_value = self.evaluate_strategy(day)
                profit = current_value - last_evaluation_value

                if profit > 0:
                    # Reinforce current strategy
                    parameters['lower_deviation'] = max(
                        parameters['lower_deviation'] * (1 - adjustment_factor), -0.1)
                    parameters['upper_deviation'] = min(
                        parameters['upper_deviation'] * (1 + adjustment_factor), 0.1)
                else:
                    # Adjust current strategy
                    parameters['lower_deviation'] = min(
                        parameters['lower_deviation'] * (1 + adjustment_factor), 0)
                    parameters['upper_deviation'] = max(
                        parameters['upper_deviation'] * (1 - adjustment_factor), 0)
                    parameters['stop_loss'] = min(
                        parameters['stop_loss'] + adjustment_factor, 0.2)  # Tighten stop loss
                    parameters['take_gain'] = max(
                        parameters['take_gain'] - adjustment_factor, 0.05)  # Decrease take gain

                last_evaluation_value = current_value

                # New logic to adjust trading strategy based on correlations
                current_date = self.dataset[self.dataset['Date']
                                            == day]['Date'].iloc[0]
                self.trade_using_correlations(current_date, self.shares_held)

            if day_index == self.total_days - 1:
                final_value = self.evaluate_strategy(day)
                print(
                    f"Final Portfolio for Generation {generation_number}: Cash: {self.cash}, Shares Held: {self.shares_held}")
                return final_value - self.initial_capital, parameters


def crossover(parent_1, parent_2):
    child = {}
    for key in parent_1:
        child[key] = random.choice([parent_1[key], parent_2[key]])
    return child


def mutate(child, mutation_rate=0.1):
    for key in child:
        if random.random() < mutation_rate:
            deviation = random.uniform(-0.1, 0.1)
            child[key] += child[key] * deviation
    return child


# Load dataset
dataset = pd.read_csv(
    'C:\\Users\\lmueller\\Desktop\\Project M\\Project M Cleaned Data.csv')

# Initialize the AI
ai = TradingAI(initial_capital=100000, dataset=dataset)
ai.reset()  # Initial reset for the AI without a generation number

# Run multiple generations with GA
generations = 100
generation_profits = []
population_size = 10
population = [{
    'lower_deviation': random.uniform(-0.1, 0),
    'upper_deviation': random.uniform(0, 0.1),
    'stop_loss': random.uniform(0.05, 0.15),
    'take_gain': random.uniform(0.05, 0.15),
    'max_allocation_per_stock': random.uniform(0.1, 0.3)
} for _ in range(population_size)]

best_profit = -float('inf')
best_parameters = None
best_portfolio = {}

for generation in range(generations):
    print(f"\nStarting Generation {generation + 1}")
    ai.reset(generation + 1)

    # Generate a random initial strategy for the generation
    initial_strategy = {
        'lower_deviation': random.uniform(-0.1, 0),
        'upper_deviation': random.uniform(0, 0.1),
        'stop_loss': random.uniform(0.05, 0.15),
        'take_gain': random.uniform(0.05, 0.15),
        'max_allocation_per_stock': random.uniform(0.1, 0.3)
    }

    # Run a single simulation for the entire generation
    profit, adjusted_parameters = ai.run_simulation(
        initial_strategy, generation + 1)
    generation_profits.append(profit)

    # Update best profit and parameters
    if profit > best_profit:
        best_profit = profit
        best_parameters = adjusted_parameters
        best_portfolio = ai.shares_held.copy()

    # Prepare for the next generation
    population = [mutate(adjusted_parameters) for _ in range(population_size)]

    print(f"Ending Generation {generation + 1} with Best Profit: {profit}")


print(
    f"\nOverall Best Generation - Profit: {best_profit}, Parameters: {best_parameters}, Portfolio: {best_portfolio}")

plt.figure(figsize=(12, 6))
plt.plot(range(1, generations + 1),
         generation_profits, marker='o', linestyle='-')
plt.title('Profit by Generation')
plt.xlabel('Generation')
plt.ylabel('Profit')
plt.grid(True)
plt.show()

# file_path = 'C:\\Users\\lmueller\\Desktop\\Project M\\Project M Cleaned Data.csv'
