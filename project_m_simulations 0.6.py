import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class TradingAI:
    def __init__(self, initial_capital, dataset_path):
        self.initial_capital = initial_capital
        self.dataset = pd.read_csv(dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        self.dataset.sort_values('Date', inplace=True)
        self.total_days = len(self.dataset['Date'].unique())
        self.correlations = self.calculate_rolling_correlations()
        self.reset()

    def reset(self, generation_number=None):
        self.cash = self.initial_capital
        self.shares_held = {company: 0 for company in self.dataset['Company'].unique()}
        self.purchase_prices = {company: 0 for company in self.dataset['Company'].unique()}
        self.total_value = self.initial_capital
        self.trade_count = 0
        if generation_number is not None:
            print(f"AI reset for Generation {generation_number}.")
        else:
            print("AI initialized with starting capital.")

    def calculate_rolling_correlations(self, window_size=30):
        pivot_df = self.dataset.pivot(index='Date', columns='Company', values='Close/Last')
        rolling_corr = pivot_df.rolling(window=window_size).corr().dropna(how='all', axis=1)
        # Aggregate correlations by taking the last available correlation matrix for the window
        corr_by_date = {date: rolling_corr.xs(date, level='Date') for date in rolling_corr.index.get_level_values('Date').unique()}
        return corr_by_date

    def trade_using_correlations(self, current_date):
        # Simplified example of how to use correlations
        if current_date in self.correlations:
            corr_matrix = self.correlations[current_date]
            for company, shares in self.shares_held.items():
                if shares > 0:
                    correlated_companies = corr_matrix[company].drop(company).sort_values(ascending=False)
                    # Example logic to sell stocks that are highly correlated with stocks that are dropping
                    for corr_company, corr_value in correlated_companies.iteritems():
                        if corr_value > 0.8:  # Arbitrary threshold for high correlation
                            if self.dataset[(self.dataset['Date'] == current_date) & (self.dataset['Company'] == corr_company)]['Close/Last'].iloc[0] < self.purchase_prices[corr_company]:
                                self.sell_stock(corr_company, self.dataset[(self.dataset['Date'] == current_date) & (self.dataset['Company'] == corr_company)]['Close/Last'].iloc[0], "Correlated Drop")

    def evaluate_strategy(self, day):
        day_data = self.dataset[self.dataset['Date'] == day]
        self.total_value = self.cash + sum(
            self.shares_held[company] * day_data[day_data['Company'] == company]['Close/Last'].iloc[0]
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
            max_investment_per_stock = self.cash * parameters['max_allocation_per_stock']

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

    def sell_stock(self, company, current_price, reason):
        self.cash += self.shares_held[company] * current_price
        self.shares_held[company] = 0

    def run_simulation(self, parameters, generation_number):
        evaluation_interval = 30
        last_evaluation_value = self.initial_capital
        adjustment_factor = 0.05

        for day_index, day in enumerate(self.dataset['Date'].unique()):
            self.trade(day, parameters)
            
            if day_index % evaluation_interval == 0 or day_index == self.total_days - 1:
                current_value = self.evaluate_strategy(day)
                profit = current_value - last_evaluation_value

                if profit > 0:
                    parameters['lower_deviation'] *= (1 + adjustment_factor)
                    parameters['upper_deviation'] *= (1 + adjustment_factor)
                else:
                    parameters['lower_deviation'] *= (1 - adjustment_factor)
                    parameters['upper_deviation'] *= (1 - adjustment_factor)

                last_evaluation_value = current_value
                self.trade_using_correlations(day)

            if day_index == self.total_days - 1:
                final_value = self.evaluate_strategy(day)
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
    'C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project M Cleaned Data.csv')

# Initialize the AI with the dataset for correlations
correlation_dataset_path = 'C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Compiled_Companies_Data_with_MA.csv'
ai = TradingAI(initial_capital=100000, dataset_path=correlation_dataset_path)
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

# file_path = 'C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Compiled_Companies_Data_with_MA.csv'
